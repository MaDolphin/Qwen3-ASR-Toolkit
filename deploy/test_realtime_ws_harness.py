#!/usr/bin/env python3
"""
Realtime WebSocket test harness for qwen3_asr_toolkit.server.

This client simulates live speech by sending float32 16 kHz mono PCM chunks at
wall-clock pace while concurrently collecting partial/segment_final/final events.
It reports realtime behavior and final WER/CER against sample reference text.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import websockets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen3_asr_toolkit.audio_tools import WAV_SAMPLE_RATE, load_audio


DEFAULT_CASES = [
    {
        "name": "sample_0",
        "audio": "sample/sample_0.mp3",
        "reference": "sample/sample_0.txt",
        "realtime": True,
    },
    {
        "name": "deutsch",
        "audio": "sample/deutsch.mp3",
        "reference": "sample/deutsch.txt",
        "realtime": True,
    },
    {
        "name": "sample_2",
        "audio": "sample/sample_2.m4a",
        "reference": "sample/sample_2.txt",
        "realtime": False,
    },
]

LANGUAGE_LABELS = {
    "chinese",
    "english",
    "german",
    "french",
    "spanish",
    "italian",
    "portuguese",
    "japanese",
    "korean",
    "russian",
}


def _edit_distance(left: List[str], right: List[str]) -> int:
    prev = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        cur = [i]
        for j, right_item in enumerate(right, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (left_item != right_item),
                )
            )
        prev = cur
    return prev[-1]


def _strip_reference_language(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().casefold() in LANGUAGE_LABELS:
        lines = lines[1:]
    return "\n".join(lines).strip()


def _normalize_words(text: str) -> List[str]:
    text = text.casefold().replace("_", " ")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return [word for word in text.split() if word]


def _normalize_chars(text: str) -> List[str]:
    text = text.casefold().replace("_", "")
    text = re.sub(r"[^\w]+", "", text, flags=re.UNICODE)
    return list(text)


def _score(reference_path: str, hypothesis: str) -> Dict:
    reference = _strip_reference_language(
        Path(reference_path).read_text(encoding="utf-8")
    )
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


def _event_latency(event: Dict, wall_elapsed_sec: float) -> Optional[float]:
    audio_duration_sec = event.get("audio_duration_sec")
    if audio_duration_sec is None:
        return None
    return round(wall_elapsed_sec - float(audio_duration_sec), 3)


async def run_case(
    *,
    ws_url: str,
    name: str,
    audio_path: str,
    reference_path: str,
    chunk_ms: int,
    decode_interval_ms: int,
    min_chunk_ms: int,
    finalize_silence_ms: int,
    max_segment_sec: float,
    realtime: bool,
    speedup: float,
    receive_timeout_sec: float,
) -> Dict:
    wav = load_audio(audio_path).astype(np.float32)
    chunk_samples = max(1, int(WAV_SAMPLE_RATE * chunk_ms / 1000.0))
    audio_duration_sec = len(wav) / WAV_SAMPLE_RATE
    send_sleep_sec = (chunk_ms / 1000.0) / max(1.0, speedup) if realtime else 0.0

    events: List[Dict] = []
    final_event: Optional[Dict] = None
    error_event: Optional[Dict] = None
    sender_done = asyncio.Event()

    async with websockets.connect(ws_url, max_size=None) as ws:
        t0 = time.perf_counter()
        ready = json.loads(await ws.recv())
        await ws.send(
            json.dumps(
                {
                    "event": "start",
                    "context": "",
                    "decode_interval_ms": decode_interval_ms,
                    "min_chunk_ms": min_chunk_ms,
                    "finalize_silence_ms": finalize_silence_ms,
                    "max_segment_sec": max_segment_sec,
                }
            )
        )
        started = json.loads(await ws.recv())
        if started.get("event") == "error":
            raise RuntimeError(started.get("message", "failed to start session"))

        async def sender():
            try:
                for pos in range(0, len(wav), chunk_samples):
                    chunk = wav[pos : pos + chunk_samples]
                    await ws.send(chunk.astype(np.float32).tobytes())
                    if send_sleep_sec > 0:
                        await asyncio.sleep(send_sleep_sec)
                await ws.send(json.dumps({"event": "finish"}))
            finally:
                sender_done.set()

        async def receiver():
            nonlocal final_event, error_event
            while True:
                timeout = receive_timeout_sec if sender_done.is_set() else None
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                now = time.perf_counter()
                event = json.loads(raw)
                event_name = event.get("event")
                if event_name in {"partial", "segment_final", "final", "error"}:
                    record = {
                        "event": event_name,
                        "wall_elapsed_sec": round(now - t0, 3),
                        "audio_duration_sec": event.get("audio_duration_sec"),
                        "realtime_lag_sec": _event_latency(event, now - t0),
                        "updated": event.get("updated"),
                        "text_len": len(event.get("text", "") or ""),
                        "language": event.get("language", "") or "",
                        "segment_index": event.get("segment_index"),
                    }
                    if event_name == "segment_final":
                        record["segment_text_len"] = len(event.get("segment_text", "") or "")
                    events.append(record)

                if event_name == "final":
                    final_event = event
                    return
                if event_name == "error":
                    error_event = event
                    return

        await asyncio.gather(sender(), receiver())

    partial_events = [event for event in events if event["event"] == "partial"]
    updated_partials = [
        event for event in partial_events if event.get("updated") is True
    ]
    nonempty_text_events = [
        event for event in events if int(event.get("text_len") or 0) > 0
    ]
    segment_events = [event for event in events if event["event"] == "segment_final"]
    final_text = final_event.get("text", "") if final_event else ""

    result = {
        "name": name,
        "audio": audio_path,
        "reference": reference_path,
        "mode": "realtime" if realtime else "accelerated",
        "speedup": 1.0 if realtime else None,
        "audio_duration_sec": round(audio_duration_sec, 3),
        "chunk_ms": chunk_ms,
        "decode_interval_ms": decode_interval_ms,
        "min_chunk_ms": min_chunk_ms,
        "finalize_silence_ms": finalize_silence_ms,
        "max_segment_sec": max_segment_sec,
        "ready": ready,
        "started": started,
        "event_counts": {
            "partial": len(partial_events),
            "partial_updated_true": len(updated_partials),
            "segment_final": len(segment_events),
            "error": 1 if error_event else 0,
        },
        "first_partial_updated_sec": (
            updated_partials[0]["wall_elapsed_sec"] if updated_partials else None
        ),
        "first_nonempty_text_sec": (
            nonempty_text_events[0]["wall_elapsed_sec"] if nonempty_text_events else None
        ),
        "final_wall_elapsed_sec": (
            next(
                (
                    event["wall_elapsed_sec"]
                    for event in events
                    if event["event"] == "final"
                ),
                None,
            )
        ),
        "max_realtime_lag_sec": (
            max(
                event["realtime_lag_sec"]
                for event in events
                if event.get("realtime_lag_sec") is not None
            )
            if realtime
            else None
        ),
        "final": final_event,
        "error": error_event,
        "metrics": _score(reference_path, final_text) if final_event else None,
        "text_preview": final_text[:300],
        "events_preview": events[:10],
        "events_tail": events[-10:],
    }
    if not realtime:
        result["note"] = "Accelerated push; realtime lag metrics are intentionally omitted."
    return result


async def main_async(args):
    cases = DEFAULT_CASES
    if args.case:
        wanted = set(args.case)
        cases = [case for case in DEFAULT_CASES if case["name"] in wanted]
    elif not args.include_long_accelerated:
        cases = [case for case in cases if case["realtime"]]

    if any(not case["realtime"] for case in cases) and not args.allow_long_accelerated:
        raise SystemExit(
            "Refusing to run accelerated long cases without --allow-long-accelerated. "
            "For sample_2, consider a larger --decode-interval-ms to avoid thousands "
            "of remote ASR calls."
        )

    results = []
    for case in cases:
        print(f"[case] {case['name']} ({'realtime' if case['realtime'] else 'accelerated'})")
        result = await run_case(
            ws_url=args.ws_url,
            name=case["name"],
            audio_path=case["audio"],
            reference_path=case["reference"],
            chunk_ms=args.chunk_ms,
            decode_interval_ms=args.decode_interval_ms,
            min_chunk_ms=args.min_chunk_ms,
            finalize_silence_ms=args.finalize_silence_ms,
            max_segment_sec=args.max_segment_sec,
            realtime=case["realtime"],
            speedup=args.speedup,
            receive_timeout_sec=args.receive_timeout_sec,
        )
        results.append(result)
        status = "ok" if result.get("final") else "error"
        metrics = result.get("metrics") or {}
        print(
            f"[case] {case['name']} {status}: "
            f"partials={result['event_counts']['partial']} "
            f"segments={result['event_counts']['segment_final']} "
            f"wer={metrics.get('wer')} cer={metrics.get('cer')}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] wrote {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime WS behavior/accuracy harness.")
    parser.add_argument(
        "--ws-url",
        default="ws://127.0.0.1:18000/ws/v1/realtime/transcribe",
        help="Realtime websocket URL.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case["name"] for case in DEFAULT_CASES],
        help="Run only the named case; repeat for multiple cases.",
    )
    parser.add_argument(
        "--include-long-accelerated",
        action="store_true",
        help="Include accelerated long cases when --case is not provided.",
    )
    parser.add_argument(
        "--allow-long-accelerated",
        action="store_true",
        help="Required to run accelerated long cases such as sample_2.",
    )
    parser.add_argument("--chunk-ms", type=int, default=500)
    parser.add_argument("--decode-interval-ms", type=int, default=600)
    parser.add_argument("--min-chunk-ms", type=int, default=200)
    parser.add_argument("--finalize-silence-ms", type=int, default=600)
    parser.add_argument("--max-segment-sec", type=float, default=20.0)
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Reserved for non-realtime cases; realtime cases always use 1x.",
    )
    parser.add_argument("--receive-timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--output",
        default="runtime/realtime_ws_report.json",
        help="JSON report path.",
    )
    return parser.parse_args()


def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
