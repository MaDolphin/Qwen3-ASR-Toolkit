#!/usr/bin/env python3
"""Validation harness for the native Qwen3-ASR streaming WebSocket server.

This client matches ``deploy/vllm_streaming_server_native.py``. It sends
float32 16 kHz mono PCM chunks, records ``started``/``ack``/``partial``/
``final``/``error`` events, and writes a structured JSON report for one audio
file.
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
        return {
            "reference_path": reference_path,
            "error": "reference file not found",
        }

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


async def run_case(
    *,
    uri: str,
    audio_path: str,
    reference_path: Optional[str],
    chunk_ms: int,
    chunk_size_sec: float,
    unfixed_chunk_num: int,
    unfixed_token_num: int,
    realtime: bool,
    speedup: float,
    receive_timeout_sec: float,
    context: str,
    language: Optional[str],
) -> Dict[str, Any]:
    wav = load_audio(audio_path).astype(np.float32)
    audio_duration_sec = len(wav) / WAV_SAMPLE_RATE
    chunk_samples = max(1, int(round(WAV_SAMPLE_RATE * chunk_ms / 1000.0)))
    send_sleep_sec = (chunk_ms / 1000.0) / max(1.0, speedup)

    events: List[Dict[str, Any]] = []
    raw_events: List[Dict[str, Any]] = []
    ack_events: List[Dict[str, Any]] = []
    partial_events: List[Dict[str, Any]] = []
    error_event: Optional[Dict[str, Any]] = None
    sender_error_event: Optional[Dict[str, Any]] = None
    started_event: Optional[Dict[str, Any]] = None
    final_event: Optional[Dict[str, Any]] = None
    sender_done = asyncio.Event()

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
        await ws.send(json.dumps(start_payload, ensure_ascii=False))
        started_raw = await asyncio.wait_for(ws.recv(), timeout=receive_timeout_sec)
        started_event = json.loads(started_raw)
        raw_events.append(
            {
                "event": started_event.get("event"),
                "wall_elapsed_sec": round(time.perf_counter() - t0, 3),
                "payload": started_event,
            }
        )
        if started_event.get("event") != "started":
            return _build_result(
                uri=uri,
                audio_path=audio_path,
                reference_path=reference_path,
                chunk_ms=chunk_ms,
                chunk_size_sec=chunk_size_sec,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                realtime=realtime,
                audio_duration_sec=audio_duration_sec,
                started_event=started_event,
                events=events,
                raw_events=raw_events,
                ack_events=ack_events,
                partial_events=partial_events,
                final_event=None,
                error_event={"event": "error", "message": "did not receive started"},
                wall_elapsed_sec=round(time.perf_counter() - t0, 3),
            )

        async def sender() -> None:
            nonlocal sender_error_event
            try:
                for pos in range(0, len(wav), chunk_samples):
                    chunk = wav[pos : pos + chunk_samples]
                    await ws.send(chunk.astype(np.float32, copy=False).tobytes())
                    if send_sleep_sec > 0:
                        await asyncio.sleep(send_sleep_sec)
                await ws.send(json.dumps({"event": "finish"}))
            except Exception as exc:
                sender_error_event = {"event": "error", "message": f"sender failed: {exc}"}
                try:
                    await ws.close()
                except Exception:
                    pass
            finally:
                sender_done.set()

        async def receiver() -> None:
            nonlocal error_event, final_event
            while True:
                timeout = receive_timeout_sec if sender_done.is_set() else None
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                wall_elapsed_sec = time.perf_counter() - t0
                event = json.loads(raw)
                event_name = event.get("event")
                raw_events.append(
                    {
                        "event": event_name,
                        "wall_elapsed_sec": round(wall_elapsed_sec, 3),
                        "payload": event,
                    }
                )

                if event_name == "ack":
                    ack_events.append(
                        {
                            "event": "ack",
                            "wall_elapsed_sec": round(wall_elapsed_sec, 3),
                            "received_samples": event.get("received_samples"),
                            "total_samples": event.get("total_samples"),
                            "duration_sec": event.get("duration_sec"),
                        }
                    )
                    continue

                if event_name == "partial":
                    record = _event_record(event, wall_elapsed_sec)
                    partial_events.append(record)
                    events.append(record)
                    continue

                if event_name == "final":
                    final_event = event
                    events.append(_event_record(event, wall_elapsed_sec))
                    return

                if event_name == "error":
                    error_event = event
                    events.append(
                        {
                            "event": "error",
                            "wall_elapsed_sec": round(wall_elapsed_sec, 3),
                            "message": event.get("message", ""),
                        }
                    )
                    return

                events.append(
                    {
                        "event": event_name,
                        "wall_elapsed_sec": round(wall_elapsed_sec, 3),
                    }
                )

        await asyncio.gather(sender(), receiver())
        if sender_error_event is not None and error_event is None and final_event is None:
            error_event = sender_error_event
            events.append({
                "event": "error",
                "wall_elapsed_sec": round(time.perf_counter() - t0, 3),
                "message": sender_error_event.get("message", ""),
            })
        wall_elapsed_sec = round(time.perf_counter() - t0, 3)

    return _build_result(
        uri=uri,
        audio_path=audio_path,
        reference_path=reference_path,
        chunk_ms=chunk_ms,
        chunk_size_sec=chunk_size_sec,
        unfixed_chunk_num=unfixed_chunk_num,
        unfixed_token_num=unfixed_token_num,
        realtime=realtime,
        audio_duration_sec=audio_duration_sec,
        started_event=started_event,
        events=events,
        raw_events=raw_events,
        ack_events=ack_events,
        partial_events=partial_events,
        final_event=final_event,
        error_event=error_event,
        wall_elapsed_sec=wall_elapsed_sec,
    )


def _build_result(
    *,
    uri: str,
    audio_path: str,
    reference_path: Optional[str],
    chunk_ms: int,
    chunk_size_sec: float,
    unfixed_chunk_num: int,
    unfixed_token_num: int,
    realtime: bool,
    audio_duration_sec: float,
    started_event: Optional[Dict[str, Any]],
    events: List[Dict[str, Any]],
    raw_events: List[Dict[str, Any]],
    ack_events: List[Dict[str, Any]],
    partial_events: List[Dict[str, Any]],
    final_event: Optional[Dict[str, Any]],
    error_event: Optional[Dict[str, Any]],
    wall_elapsed_sec: float,
) -> Dict[str, Any]:
    nonempty_partials = [event for event in partial_events if event.get("text_nonempty")]
    final_text = final_event.get("text", "") if final_event else ""
    final_record = next((event for event in events if event.get("event") == "final"), None)
    first_partial = partial_events[0] if partial_events else None
    first_nonempty_partial = nonempty_partials[0] if nonempty_partials else None
    passed = (
        started_event is not None
        and started_event.get("event") == "started"
        and bool(ack_events)
        and bool(partial_events)
        and final_event is not None
        and bool(final_text.strip())
        and error_event is None
    )

    return {
        "schema_version": 1,
        "created_at": _utc_now(),
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
            "speedup": speedup,
            "chunk_ms": chunk_ms,
            "chunk_size_sec": chunk_size_sec,
            "unfixed_chunk_num": unfixed_chunk_num,
            "unfixed_token_num": unfixed_token_num,
            "sample_rate": WAV_SAMPLE_RATE,
        },
        "started": started_event,
        "audio_duration_sec": round(audio_duration_sec, 3),
        "wall_elapsed_sec": wall_elapsed_sec,
        "event_counts": {
            "ack": len(ack_events),
            "partial": len(partial_events),
            "partial_nonempty": len(nonempty_partials),
            "final": 1 if final_event else 0,
            "error": 1 if error_event else 0,
        },
        "latency": {
            "first_partial_sec": first_partial.get("wall_elapsed_sec") if first_partial else None,
            "first_nonempty_partial_sec": (
                first_nonempty_partial.get("wall_elapsed_sec")
                if first_nonempty_partial
                else None
            ),
            "final_sec": final_record.get("wall_elapsed_sec") if final_record else None,
        },
        "final": {
            "language": final_event.get("language", "") if final_event else "",
            "chunk_id": final_event.get("chunk_id") if final_event else None,
            "text_len": len(final_text),
            "text": final_text,
        },
        "score": _score(reference_path, final_text),
        "passed": passed,
        "error": error_event,
        "ack_events": ack_events,
        "partial_events": partial_events,
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
    parser.add_argument("--chunk-ms", type=int, default=500)
    parser.add_argument("--chunk-size-sec", type=float, default=1.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--speedup", type=float, default=1.0, help="Audio send speedup when accelerated.")
    parser.add_argument("--receive-timeout-sec", type=float, default=300.0)
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
                chunk_ms=args.chunk_ms,
                chunk_size_sec=args.chunk_size_sec,
                unfixed_chunk_num=args.unfixed_chunk_num,
                unfixed_token_num=args.unfixed_token_num,
                realtime=realtime,
                speedup=speedup,
                receive_timeout_sec=args.receive_timeout_sec,
                context=args.context,
                language=args.language,
            )
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "created_at": _utc_now(),
            "config": {
                "uri": args.uri,
                "audio": args.input,
                "reference": args.reference,
                "mode": "realtime" if realtime else "accelerated",
                "speedup": speedup,
                "chunk_ms": args.chunk_ms,
                "chunk_size_sec": args.chunk_size_sec,
                "unfixed_chunk_num": args.unfixed_chunk_num,
                "unfixed_token_num": args.unfixed_token_num,
                "sample_rate": WAV_SAMPLE_RATE,
            },
            "passed": False,
            "error": {"event": "error", "message": str(exc), "type": type(exc).__name__},
        }

    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("passed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
