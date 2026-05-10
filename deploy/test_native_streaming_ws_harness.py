#!/usr/bin/env python3
"""Validation harness for the native Qwen3-ASR streaming WebSocket server.

The client matches ``deploy/vllm_streaming_server_native.py``. It sends 16 kHz
mono float32 PCM chunks over WebSocket, records native ``started``/``ack``/
``partial``/``final``/``error`` events, and writes a structured JSON result.

For long-audio validation it supports in-memory slicing and client-side ack
window flow control so realtime sends do not build unbounded WebSocket buffers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

import numpy as np
import websockets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen3_asr_toolkit.audio_tools import WAV_SAMPLE_RATE, load_audio


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_words(text: str) -> List[str]:
    text = text.casefold().replace("_", " ")
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return [word for word in text.split() if word]


def _normalize_chars(text: str) -> List[str]:
    text = text.casefold().replace("_", "")
    text = re.sub(r"[^\w]+", "", text, flags=re.UNICODE)
    return list(text)


def _edit_distance(left: List[str], right: List[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _strip_reference_language(text: str) -> str:
    first_line, _, rest = text.partition("\n")
    if first_line.strip().lower().startswith("language:"):
        return rest.strip()
    return text.strip()


def _score(reference_path: Optional[str], hypothesis: str) -> Optional[Dict[str, Any]]:
    if not reference_path:
        return None
    path = Path(reference_path)
    if not path.exists():
        return {"reference_path": reference_path, "error": "reference file not found"}

    reference = _strip_reference_language(path.read_text(encoding="utf-8"))
    ref_words = _normalize_words(reference)
    hyp_words = _normalize_words(hypothesis)
    ref_chars = _normalize_chars(reference)
    hyp_chars = _normalize_chars(hypothesis)
    return {
        "reference_path": reference_path,
        "reference_words": len(ref_words),
        "hypothesis_words": len(hyp_words),
        "reference_chars": len(ref_chars),
        "hypothesis_chars": len(hyp_chars),
        "wer": round(_edit_distance(ref_words, hyp_words) / max(1, len(ref_words)), 4),
        "cer": round(_edit_distance(ref_chars, hyp_chars) / max(1, len(ref_chars)), 4),
    }


def _run_command(args: List[str]) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {
            "command": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"command": args, "error": str(exc)}


def _gpu_snapshot() -> Dict[str, Any]:
    return _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )


def _event_record(event: Dict[str, Any], wall_elapsed_sec: float) -> Dict[str, Any]:
    text = event.get("text", "") or ""
    return {
        "event": event.get("event"),
        "wall_elapsed_sec": round(wall_elapsed_sec, 3),
        "chunk_id": event.get("chunk_id"),
        "text_len": len(text),
        "text_nonempty": bool(text.strip()),
        "language": event.get("language", "") or "",
    }


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


class EventJsonlWriter:
    def __init__(self, path: Optional[str]) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def write(self, payload: Dict[str, Any]) -> None:
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


async def run_case(
    *,
    uri: str,
    audio_path: str,
    reference_path: Optional[str],
    output_path: Optional[str],
    event_jsonl: Optional[str],
    case_label: Optional[str],
    start_sec: float,
    duration_sec: Optional[float],
    chunk_ms: int,
    chunk_size_sec: float,
    unfixed_chunk_num: int,
    unfixed_token_num: int,
    realtime: bool,
    speedup: float,
    receive_timeout_sec: float,
    send_timeout_sec: float,
    ack_timeout_sec: float,
    max_inflight_chunks: int,
    context: str,
    language: Optional[str],
) -> Dict[str, Any]:
    full_wav = load_audio(audio_path).astype(np.float32)
    source_audio_duration_sec = len(full_wav) / WAV_SAMPLE_RATE
    start_sample = max(0, min(len(full_wav), int(round(start_sec * WAV_SAMPLE_RATE))))
    if duration_sec is None:
        end_sample = len(full_wav)
    else:
        end_sample = min(len(full_wav), start_sample + int(round(duration_sec * WAV_SAMPLE_RATE)))
    wav = full_wav[start_sample:end_sample]
    sent_duration_sec = len(wav) / WAV_SAMPLE_RATE
    chunk_samples = max(1, int(round(WAV_SAMPLE_RATE * chunk_ms / 1000.0)))
    send_interval_sec = (chunk_ms / 1000.0) / max(1.0, speedup)
    mode = "realtime" if realtime else "accelerated"
    case_label = case_label or Path(audio_path).stem
    max_inflight_chunks = max(1, max_inflight_chunks)

    events: List[Dict[str, Any]] = []
    raw_events: List[Dict[str, Any]] = []
    ack_events: List[Dict[str, Any]] = []
    partial_events: List[Dict[str, Any]] = []
    send_events: List[Dict[str, Any]] = []
    send_stall_events: List[Dict[str, Any]] = []
    error_events: List[Dict[str, Any]] = []
    client_error: Optional[Dict[str, Any]] = None
    started_event: Optional[Dict[str, Any]] = None
    final_event: Optional[Dict[str, Any]] = None
    final_record: Optional[Dict[str, Any]] = None
    chunks_sent = 0
    acks_received = 0
    max_observed_inflight_chunks = 0
    max_realtime_lag_sec = 0.0
    send_stall_total_sec = 0.0
    sender_done = asyncio.Event()
    ack_condition = asyncio.Condition()
    writer = EventJsonlWriter(event_jsonl)

    def wall(t0: float) -> float:
        return time.perf_counter() - t0

    def inflight() -> int:
        return chunks_sent - acks_received

    def log_jsonl(t0: float, direction: str, event_name: str, **extra: Any) -> None:
        writer.write(
            {
                "created_at": _utc_now(),
                "wall_elapsed_sec": round(wall(t0), 3),
                "direction": direction,
                "event": event_name,
                "chunks_sent": chunks_sent,
                "acks_received": acks_received,
                "inflight_chunks": inflight(),
                **extra,
            }
        )

    start_payload: Dict[str, Any] = {
        "event": "start",
        "stream": True,
        "context": context,
        "chunk_size_sec": chunk_size_sec,
        "unfixed_chunk_num": unfixed_chunk_num,
        "unfixed_token_num": unfixed_token_num,
    }
    if language:
        start_payload["language"] = language

    async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
        t0 = time.perf_counter()
        await asyncio.wait_for(
            ws.send(json.dumps(start_payload, ensure_ascii=False)), timeout=send_timeout_sec
        )
        log_jsonl(t0, "send", "start")
        started_raw = await asyncio.wait_for(ws.recv(), timeout=receive_timeout_sec)
        started_event = json.loads(started_raw)
        raw_events.append(
            {
                "event": started_event.get("event"),
                "wall_elapsed_sec": round(wall(t0), 3),
                "payload": started_event,
            }
        )
        log_jsonl(t0, "recv", str(started_event.get("event")), payload=started_event)
        if started_event.get("event") != "started":
            return _build_result(
                uri=uri,
                audio_path=audio_path,
                reference_path=reference_path,
                case_label=case_label,
                start_sec=start_sec,
                requested_duration_sec=duration_sec,
                source_audio_duration_sec=source_audio_duration_sec,
                sent_duration_sec=sent_duration_sec,
                chunk_ms=chunk_ms,
                chunk_size_sec=chunk_size_sec,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                realtime=realtime,
                speedup=speedup,
                max_inflight_chunks=max_inflight_chunks,
                max_observed_inflight_chunks=max_observed_inflight_chunks,
                max_realtime_lag_sec=max_realtime_lag_sec,
                send_stall_events=send_stall_events,
                send_stall_total_sec=send_stall_total_sec,
                chunks_sent=chunks_sent,
                started_event=started_event,
                events=events,
                raw_events=raw_events,
                send_events=send_events,
                ack_events=ack_events,
                partial_events=partial_events,
                final_event=None,
                final_record=None,
                error_events=error_events,
                client_error={"event": "error", "message": "did not receive started"},
                wall_elapsed_sec=round(wall(t0), 3),
                output_path=output_path,
                event_jsonl=event_jsonl,
            )

        async def wait_for_window() -> None:
            nonlocal client_error, send_stall_total_sec, max_realtime_lag_sec
            async with ack_condition:
                if inflight() < max_inflight_chunks:
                    return
                stall_start = time.perf_counter()
                send_stall_events.append(
                    {
                        "wall_elapsed_sec": round(wall(t0), 3),
                        "inflight_chunks": inflight(),
                        "max_inflight_chunks": max_inflight_chunks,
                    }
                )
                log_jsonl(t0, "client", "send_stall_start", max_inflight_chunks=max_inflight_chunks)
                try:
                    await asyncio.wait_for(
                        ack_condition.wait_for(lambda: inflight() < max_inflight_chunks),
                        timeout=ack_timeout_sec,
                    )
                except asyncio.TimeoutError:
                    stalled_sec = time.perf_counter() - stall_start
                    send_stall_total_sec += stalled_sec
                    max_realtime_lag_sec = max(max_realtime_lag_sec, stalled_sec)
                    send_stall_events[-1]["stall_sec"] = round(stalled_sec, 3)
                    send_stall_events[-1]["timeout"] = True
                    send_stall_events[-1]["resolved_wall_elapsed_sec"] = round(wall(t0), 3)
                    client_error = {
                        "event": "error",
                        "message": f"ack timeout after {ack_timeout_sec}s",
                        "type": "AckTimeout",
                    }
                    log_jsonl(
                        t0,
                        "client",
                        "ack_timeout",
                        max_inflight_chunks=max_inflight_chunks,
                        stall_sec=round(stalled_sec, 3),
                    )
                    raise
                stalled_sec = time.perf_counter() - stall_start
                send_stall_total_sec += stalled_sec
                max_realtime_lag_sec = max(max_realtime_lag_sec, stalled_sec)
                send_stall_events[-1]["stall_sec"] = round(stalled_sec, 3)
                send_stall_events[-1]["resolved_wall_elapsed_sec"] = round(wall(t0), 3)
                log_jsonl(t0, "client", "send_stall_end", stall_sec=round(stalled_sec, 3))

        async def sender() -> None:
            nonlocal chunks_sent, max_observed_inflight_chunks, max_realtime_lag_sec, client_error
            try:
                next_send_at = time.perf_counter()
                for chunk_index, pos in enumerate(range(0, len(wav), chunk_samples), start=1):
                    if realtime:
                        now = time.perf_counter()
                        if now < next_send_at:
                            await asyncio.sleep(next_send_at - now)
                    await wait_for_window()

                    planned_lag = max(0.0, time.perf_counter() - next_send_at) if realtime else 0.0
                    max_realtime_lag_sec = max(max_realtime_lag_sec, planned_lag)
                    chunk = wav[pos : pos + chunk_samples]
                    await asyncio.wait_for(
                        ws.send(chunk.astype(np.float32, copy=False).tobytes()),
                        timeout=send_timeout_sec,
                    )
                    chunks_sent += 1
                    max_observed_inflight_chunks = max(max_observed_inflight_chunks, inflight())
                    sent_record = {
                        "event": "audio",
                        "wall_elapsed_sec": round(wall(t0), 3),
                        "chunk_index": chunk_index,
                        "samples": int(chunk.size),
                        "audio_offset_sec": round(pos / WAV_SAMPLE_RATE, 3),
                        "inflight_chunks": inflight(),
                        "realtime_lag_sec": round(planned_lag, 3),
                    }
                    send_events.append(sent_record)
                    log_jsonl(t0, "send", "audio", send_event=sent_record)
                    if realtime:
                        next_send_at = max(next_send_at + send_interval_sec, time.perf_counter() + send_interval_sec)

                async with ack_condition:
                    try:
                        await asyncio.wait_for(
                            ack_condition.wait_for(lambda: acks_received >= chunks_sent),
                            timeout=ack_timeout_sec,
                        )
                    except asyncio.TimeoutError:
                        client_error = {
                            "event": "error",
                            "message": f"final ack timeout after {ack_timeout_sec}s",
                            "type": "AckTimeout",
                        }
                        log_jsonl(t0, "client", "final_ack_timeout")
                        raise
                await asyncio.wait_for(
                    ws.send(json.dumps({"event": "finish"}, ensure_ascii=False)),
                    timeout=send_timeout_sec,
                )
                log_jsonl(t0, "send", "finish")
            except Exception as exc:
                if client_error is None:
                    client_error = {
                        "event": "error",
                        "message": f"sender failed: {exc}",
                        "type": type(exc).__name__,
                    }
                try:
                    await ws.close()
                except Exception:
                    pass
            finally:
                sender_done.set()

        async def receiver() -> None:
            nonlocal acks_received, error_events, final_event, final_record
            while True:
                timeout = receive_timeout_sec if sender_done.is_set() else None
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                wall_elapsed_sec = wall(t0)
                event = json.loads(raw)
                event_name = event.get("event")
                raw_events.append(
                    {
                        "event": event_name,
                        "wall_elapsed_sec": round(wall_elapsed_sec, 3),
                        "payload": event,
                    }
                )
                log_jsonl(t0, "recv", str(event_name), payload=event)

                if event_name == "ack":
                    acks_received += 1
                    ack_events.append(
                        {
                            "event": "ack",
                            "wall_elapsed_sec": round(wall_elapsed_sec, 3),
                            "received_samples": event.get("received_samples"),
                            "total_samples": event.get("total_samples"),
                            "duration_sec": event.get("duration_sec"),
                            "inflight_chunks": inflight(),
                        }
                    )
                    async with ack_condition:
                        ack_condition.notify_all()
                    continue

                if event_name == "partial":
                    record = _event_record(event, wall_elapsed_sec)
                    partial_events.append(record)
                    events.append(record)
                    continue

                if event_name == "final":
                    final_event = event
                    final_record = _event_record(event, wall_elapsed_sec)
                    events.append(final_record)
                    return

                if event_name == "error":
                    error_events.append(event)
                    events.append(
                        {
                            "event": "error",
                            "wall_elapsed_sec": round(wall_elapsed_sec, 3),
                            "message": event.get("message"),
                        }
                    )
                    return

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        await asyncio.wait({sender_task, receiver_task}, return_when=asyncio.FIRST_COMPLETED)
        if client_error is not None and not receiver_task.done():
            receiver_task.cancel()
        await asyncio.gather(sender_task, receiver_task, return_exceptions=True)

        return _build_result(
            uri=uri,
            audio_path=audio_path,
            reference_path=reference_path,
            case_label=case_label,
            start_sec=start_sec,
            requested_duration_sec=duration_sec,
            source_audio_duration_sec=source_audio_duration_sec,
            sent_duration_sec=sent_duration_sec,
            chunk_ms=chunk_ms,
            chunk_size_sec=chunk_size_sec,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            realtime=realtime,
            speedup=speedup,
            max_inflight_chunks=max_inflight_chunks,
            max_observed_inflight_chunks=max_observed_inflight_chunks,
            max_realtime_lag_sec=max_realtime_lag_sec,
            send_stall_events=send_stall_events,
            send_stall_total_sec=send_stall_total_sec,
            chunks_sent=chunks_sent,
            started_event=started_event,
            events=events,
            raw_events=raw_events,
            send_events=send_events,
            ack_events=ack_events,
            partial_events=partial_events,
            final_event=final_event,
            final_record=final_record,
            error_events=error_events,
            client_error=client_error,
            wall_elapsed_sec=round(wall(t0), 3),
            output_path=output_path,
            event_jsonl=event_jsonl,
        )


def _build_result(
    *,
    uri: str,
    audio_path: str,
    reference_path: Optional[str],
    case_label: str,
    start_sec: float,
    requested_duration_sec: Optional[float],
    source_audio_duration_sec: float,
    sent_duration_sec: float,
    chunk_ms: int,
    chunk_size_sec: float,
    unfixed_chunk_num: int,
    unfixed_token_num: int,
    realtime: bool,
    speedup: float,
    max_inflight_chunks: int,
    max_observed_inflight_chunks: int,
    max_realtime_lag_sec: float,
    send_stall_events: List[Dict[str, Any]],
    send_stall_total_sec: float,
    chunks_sent: int,
    started_event: Optional[Dict[str, Any]],
    events: List[Dict[str, Any]],
    raw_events: List[Dict[str, Any]],
    send_events: List[Dict[str, Any]],
    ack_events: List[Dict[str, Any]],
    partial_events: List[Dict[str, Any]],
    final_event: Optional[Dict[str, Any]],
    final_record: Optional[Dict[str, Any]],
    error_events: List[Dict[str, Any]],
    client_error: Optional[Dict[str, Any]],
    wall_elapsed_sec: float,
    output_path: Optional[str],
    event_jsonl: Optional[str],
) -> Dict[str, Any]:
    nonempty_partials = [event for event in partial_events if event.get("text_nonempty")]
    final_text = final_event.get("text", "") if final_event else ""
    first_ack = ack_events[0] if ack_events else None
    last_ack = ack_events[-1] if ack_events else None
    first_partial = partial_events[0] if partial_events else None
    first_nonempty_partial = nonempty_partials[0] if nonempty_partials else None
    partial_times = [float(event["wall_elapsed_sec"]) for event in partial_events]
    partial_intervals = [
        round(right - left, 3) for left, right in zip(partial_times, partial_times[1:])
    ]
    error_count = len(error_events)
    passed = (
        started_event is not None
        and started_event.get("event") == "started"
        and chunks_sent > 0
        and len(ack_events) == chunks_sent
        and bool(partial_events)
        and final_event is not None
        and bool(final_text.strip())
        and error_count == 0
        and client_error is None
    )

    return {
        "schema_version": 2,
        "created_at": _utc_now(),
        "case_label": case_label,
        "environment": {
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "gpu_snapshot_after_case": _gpu_snapshot(),
        },
        "config": {
            "uri": uri,
            "audio": audio_path,
            "reference": reference_path,
            "mode": "realtime" if realtime else "accelerated",
            "send_mode": "realtime" if realtime else "accelerated",
            "speedup": speedup,
            "chunk_ms": chunk_ms,
            "chunk_size_sec": chunk_size_sec,
            "unfixed_chunk_num": unfixed_chunk_num,
            "unfixed_token_num": unfixed_token_num,
            "sample_rate": WAV_SAMPLE_RATE,
            "start_sec": round(start_sec, 3),
            "requested_duration_sec": round(requested_duration_sec, 3) if requested_duration_sec is not None else None,
            "source_audio_duration_sec": round(source_audio_duration_sec, 3),
            "sent_duration_sec": round(sent_duration_sec, 3),
            "max_inflight_chunks": max_inflight_chunks,
            "output": output_path,
            "event_jsonl": event_jsonl,
        },
        "started": started_event,
        "audio_duration_sec": round(sent_duration_sec, 3),
        "source_audio_duration_sec": round(source_audio_duration_sec, 3),
        "start_sec": round(start_sec, 3),
        "requested_duration_sec": round(requested_duration_sec, 3) if requested_duration_sec is not None else None,
        "sent_duration_sec": round(sent_duration_sec, 3),
        "wall_elapsed_sec": wall_elapsed_sec,
        "event_counts": {
            "chunks_sent": chunks_sent,
            "ack": len(ack_events),
            "partial": len(partial_events),
            "partial_nonempty": len(nonempty_partials),
            "final": 1 if final_event else 0,
            "error": error_count,
        },
        "counts": {
            "chunks_sent": chunks_sent,
            "ack_count": len(ack_events),
            "partial_count": len(partial_events),
            "error_count": error_count,
        },
        "flow_control": {
            "max_inflight_chunks": max_inflight_chunks,
            "max_observed_inflight_chunks": max_observed_inflight_chunks,
            "send_stall_count": len(send_stall_events),
            "send_stall_total_sec": round(send_stall_total_sec, 3),
            "max_realtime_lag_sec": round(max_realtime_lag_sec, 3),
            "send_stall_events": send_stall_events,
        },
        "latency": {
            "first_ack_sec": first_ack.get("wall_elapsed_sec") if first_ack else None,
            "last_ack_sec": last_ack.get("wall_elapsed_sec") if last_ack else None,
            "first_partial_sec": first_partial.get("wall_elapsed_sec") if first_partial else None,
            "first_nonempty_partial_sec": (
                first_nonempty_partial.get("wall_elapsed_sec") if first_nonempty_partial else None
            ),
            "final_sec": final_record.get("wall_elapsed_sec") if final_record else None,
            "partial_interval_p50_sec": round(median(partial_intervals), 3) if partial_intervals else None,
            "partial_interval_p95_sec": _percentile(partial_intervals, 0.95),
        },
        "timing": {
            "wall_elapsed_sec": wall_elapsed_sec,
            "first_ack_sec": first_ack.get("wall_elapsed_sec") if first_ack else None,
            "last_ack_sec": last_ack.get("wall_elapsed_sec") if last_ack else None,
            "first_partial_sec": first_partial.get("wall_elapsed_sec") if first_partial else None,
            "first_nonempty_partial_sec": (
                first_nonempty_partial.get("wall_elapsed_sec") if first_nonempty_partial else None
            ),
            "final_sec": final_record.get("wall_elapsed_sec") if final_record else None,
            "partial_interval_p50_sec": round(median(partial_intervals), 3) if partial_intervals else None,
            "partial_interval_p95_sec": _percentile(partial_intervals, 0.95),
        },
        "final": {
            "language": final_event.get("language", "") if final_event else "",
            "chunk_id": final_event.get("chunk_id") if final_event else None,
            "text_len": len(final_text),
            "text": final_text,
        },
        "score": _score(reference_path, final_text),
        "passed": passed,
        "error": error_events[0] if error_events else None,
        "errors": error_events,
        "client_error": client_error,
        "ack_events": ack_events,
        "partial_events": partial_events,
        "send_events": send_events,
        "events": events,
        "raw_events": raw_events,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate native Qwen3-ASR WebSocket streaming on one audio file."
    )
    parser.add_argument("--uri", default="ws://127.0.0.1:10012/ws/stream")
    parser.add_argument("--input", required=True, help="Input audio path.")
    parser.add_argument("--reference", default=None, help="Optional reference text path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--event-jsonl", default=None, help="Optional per-event JSONL path.")
    parser.add_argument("--case-label", default=None)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--chunk-ms", type=int, default=500)
    parser.add_argument("--chunk-size-sec", type=float, default=1.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--speedup", type=float, default=1.0, help="Audio send speedup when accelerated.")
    parser.add_argument("--receive-timeout-sec", type=float, default=300.0)
    parser.add_argument("--send-timeout-sec", type=float, default=30.0)
    parser.add_argument("--ack-timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-inflight-chunks", type=int, default=4)
    parser.add_argument("--context", default="")
    parser.add_argument("--language", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--realtime", action="store_true", help="Sleep between chunks.")
    mode.add_argument("--accelerated", action="store_true", help="Send chunks without sleep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    realtime = not args.accelerated
    speedup = 1.0 if realtime else args.speedup
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = asyncio.run(
            run_case(
                uri=args.uri,
                audio_path=args.input,
                reference_path=args.reference,
                output_path=args.output,
                event_jsonl=args.event_jsonl,
                case_label=args.case_label,
                start_sec=args.start_sec,
                duration_sec=args.duration_sec,
                chunk_ms=args.chunk_ms,
                chunk_size_sec=args.chunk_size_sec,
                unfixed_chunk_num=args.unfixed_chunk_num,
                unfixed_token_num=args.unfixed_token_num,
                realtime=realtime,
                speedup=speedup,
                receive_timeout_sec=args.receive_timeout_sec,
                send_timeout_sec=args.send_timeout_sec,
                ack_timeout_sec=args.ack_timeout_sec,
                max_inflight_chunks=args.max_inflight_chunks,
                context=args.context,
                language=args.language,
            )
        )
    except Exception as exc:
        result = {
            "schema_version": 2,
            "created_at": _utc_now(),
            "case_label": args.case_label,
            "config": {
                "uri": args.uri,
                "audio": args.input,
                "reference": args.reference,
                "mode": "realtime" if realtime else "accelerated",
                "send_mode": "realtime" if realtime else "accelerated",
                "speedup": speedup,
                "chunk_ms": args.chunk_ms,
                "chunk_size_sec": args.chunk_size_sec,
                "unfixed_chunk_num": args.unfixed_chunk_num,
                "unfixed_token_num": args.unfixed_token_num,
                "sample_rate": WAV_SAMPLE_RATE,
                "start_sec": args.start_sec,
                "requested_duration_sec": args.duration_sec,
                "max_inflight_chunks": args.max_inflight_chunks,
                "output": args.output,
                "event_jsonl": args.event_jsonl,
            },
            "passed": False,
            "client_error": {"event": "error", "message": str(exc), "type": type(exc).__name__},
            "error": {"event": "error", "message": str(exc), "type": type(exc).__name__},
        }

    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("passed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
