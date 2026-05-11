#!/usr/bin/env python3
"""Windowed realtime validation harness for native Qwen3-ASR WebSocket streaming.

This harness is intended for long audio that exceeds the stable duration of a
single native streaming WebSocket session. It splits the source audio into
sequential in-memory windows and runs each window through the existing native
WebSocket protocol as an independent ``start -> audio -> finish -> final``
session. The window finals are then aggregated into one JSON and one Markdown
report.

This is a service-layer rolling-window strategy. It does not prove KV-cache true
incremental streaming because the backend still decodes each session with the
current cumulative generate implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.validation.test_native_streaming_ws_harness import run_case
from qwen3_asr_toolkit.audio_tools import WAV_SAMPLE_RATE, load_audio


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_command(args: List[str], timeout_sec: float = 15.0) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
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


def _short_text(text: str, limit: int = 300) -> str:
    text = (text or "").strip().replace("`", "'")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _strip_reference_language(text: str) -> str:
    first_line, _, rest = text.partition("\n")
    if first_line.strip().lower().startswith("language:"):
        return rest.strip()
    return text.strip()


def _window_plan(total_duration_sec: float, window_sec: float, overlap_sec: float) -> List[Dict[str, float]]:
    if window_sec <= 0:
        raise ValueError("window_sec must be > 0")
    if overlap_sec < 0:
        raise ValueError("overlap_sec must be >= 0")
    if overlap_sec >= window_sec:
        raise ValueError("overlap_sec must be smaller than window_sec")

    windows: List[Dict[str, float]] = []
    start_sec = 0.0
    index = 1
    step_sec = window_sec - overlap_sec
    while start_sec < total_duration_sec:
        remaining = max(0.0, total_duration_sec - start_sec)
        duration_sec = min(window_sec, remaining)
        windows.append(
            {
                "window_index": index,
                "start_sec": round(start_sec, 3),
                "duration_sec": round(duration_sec, 3),
                "end_sec": round(start_sec + duration_sec, 3),
            }
        )
        if start_sec + duration_sec >= total_duration_sec:
            break
        start_sec += step_sec
        index += 1
    return windows


def _case_summary(case: Dict[str, Any]) -> Dict[str, Any]:
    counts = case.get("counts") or case.get("event_counts") or {}
    latency = case.get("latency") or case.get("timing") or {}
    final = case.get("final") or {}
    flow = case.get("flow_control") or {}
    return {
        "case_label": case.get("case_label"),
        "passed": bool(case.get("passed")),
        "start_sec": case.get("start_sec"),
        "sent_duration_sec": case.get("sent_duration_sec"),
        "wall_elapsed_sec": case.get("wall_elapsed_sec"),
        "chunks_sent": counts.get("chunks_sent"),
        "ack_count": counts.get("ack_count", counts.get("ack")),
        "partial_count": counts.get("partial_count", counts.get("partial")),
        "first_partial_sec": latency.get("first_partial_sec"),
        "final_sec": latency.get("final_sec"),
        "final_text_len": final.get("text_len"),
        "language": final.get("language", ""),
        "max_observed_inflight_chunks": flow.get("max_observed_inflight_chunks"),
        "send_stall_count": flow.get("send_stall_count"),
        "send_stall_total_sec": flow.get("send_stall_total_sec"),
        "max_realtime_lag_sec": flow.get("max_realtime_lag_sec"),
        "client_error": (case.get("client_error") or {}).get("message"),
        "server_error": (case.get("error") or {}).get("message"),
        "final_excerpt": _short_text(final.get("text", ""), 180),
    }


def _aggregate_text(cases: List[Dict[str, Any]], separator: str) -> str:
    return separator.join(
        (case.get("final") or {}).get("text", "").strip()
        for case in cases
        if (case.get("final") or {}).get("text", "").strip()
    )


def _build_result(
    *,
    args: argparse.Namespace,
    source_audio_duration_sec: float,
    windows: List[Dict[str, float]],
    cases: List[Dict[str, Any]],
    started_at: str,
    wall_elapsed_sec: float,
    stopped_early: bool,
    gpu_before: Dict[str, Any],
    gpu_after: Dict[str, Any],
) -> Dict[str, Any]:
    summaries = [_case_summary(case) for case in cases]
    aggregate_text = _aggregate_text(cases, args.text_separator)
    passed = bool(cases) and len(cases) == len(windows) and all(case.get("passed") for case in cases)
    total_sent_duration_sec = round(sum(float(case.get("sent_duration_sec") or 0) for case in cases), 3)
    total_chunks_sent = sum(int((case.get("counts") or {}).get("chunks_sent") or 0) for case in cases)
    total_ack_count = sum(int((case.get("counts") or {}).get("ack_count") or 0) for case in cases)
    total_partial_count = sum(int((case.get("counts") or {}).get("partial_count") or 0) for case in cases)
    max_realtime_lag_sec = max(
        [float((case.get("flow_control") or {}).get("max_realtime_lag_sec") or 0) for case in cases]
        or [0.0]
    )
    total_stall_count = sum(int((case.get("flow_control") or {}).get("send_stall_count") or 0) for case in cases)
    total_stall_sec = round(
        sum(float((case.get("flow_control") or {}).get("send_stall_total_sec") or 0) for case in cases),
        3,
    )
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "started_at": started_at,
        "case_label": args.case_label,
        "passed": passed,
        "stopped_early": stopped_early,
        "environment": {
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
        },
        "config": {
            "uri": args.uri,
            "audio": args.input,
            "reference": args.reference,
            "output": args.output,
            "report": args.report,
            "case_dir": args.case_dir,
            "case_label": args.case_label,
            "window_sec": args.window_sec,
            "overlap_sec": args.overlap_sec,
            "max_windows": args.max_windows,
            "stop_on_failure": args.stop_on_failure,
            "chunk_ms": args.chunk_ms,
            "chunk_size_sec": args.chunk_size_sec,
            "unfixed_chunk_num": args.unfixed_chunk_num,
            "unfixed_token_num": args.unfixed_token_num,
            "max_inflight_chunks": args.max_inflight_chunks,
            "send_timeout_sec": args.send_timeout_sec,
            "ack_timeout_sec": args.ack_timeout_sec,
            "receive_timeout_sec": args.receive_timeout_sec,
            "sample_rate": WAV_SAMPLE_RATE,
            "mode": "realtime",
        },
        "source_audio_duration_sec": round(source_audio_duration_sec, 3),
        "window_count_planned": len(windows),
        "window_count_completed": len(cases),
        "windows": windows,
        "wall_elapsed_sec": round(wall_elapsed_sec, 3),
        "totals": {
            "sent_duration_sec": total_sent_duration_sec,
            "chunks_sent": total_chunks_sent,
            "ack_count": total_ack_count,
            "partial_count": total_partial_count,
            "aggregate_text_len": len(aggregate_text),
            "send_stall_count": total_stall_count,
            "send_stall_total_sec": total_stall_sec,
            "max_realtime_lag_sec": round(max_realtime_lag_sec, 3),
        },
        "window_summaries": summaries,
        "aggregate_text": aggregate_text,
        "cases": cases,
    }


def _write_markdown_report(result: Dict[str, Any], report_path: str) -> None:
    config = result.get("config", {})
    totals = result.get("totals", {})
    lines: List[str] = []
    lines.append("# Sample 2 Windowed Realtime Streaming Report")
    lines.append("")
    lines.append(f"- Generated at: `{_utc_now()}`")
    lines.append(f"- Audio: `{config.get('audio')}`")
    lines.append(f"- Source duration: `{result.get('source_audio_duration_sec')}` seconds")
    lines.append(f"- Window: `{config.get('window_sec')}` seconds, overlap `{config.get('overlap_sec')}` seconds")
    lines.append(f"- Overall passed: `{'yes' if result.get('passed') else 'no'}`")
    lines.append(f"- Wall elapsed: `{result.get('wall_elapsed_sec')}` seconds")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Planned windows: `{result.get('window_count_planned')}`")
    lines.append(f"- Completed windows: `{result.get('window_count_completed')}`")
    lines.append(f"- Sent duration total: `{totals.get('sent_duration_sec')}` seconds")
    lines.append(f"- Chunks/ack/partial: `{totals.get('chunks_sent')}` / `{totals.get('ack_count')}` / `{totals.get('partial_count')}`")
    lines.append(f"- Aggregate text length: `{totals.get('aggregate_text_len')}`")
    lines.append(f"- Send stalls: `{totals.get('send_stall_count')}`, stall total `{totals.get('send_stall_total_sec')}` seconds")
    lines.append(f"- Max realtime lag: `{totals.get('max_realtime_lag_sec')}` seconds")
    lines.append("")
    lines.append("## Window Results")
    lines.append("| window | start(s) | duration(s) | wall(s) | chunks | ack | partial | first partial(s) | final(s) | final len | lang | stalls | max lag(s) | passed | error |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|")
    for index, item in enumerate(result.get("window_summaries", []), start=1):
        error = item.get("client_error") or item.get("server_error") or ""
        lines.append(
            "| {index} | {start_sec} | {sent_duration_sec} | {wall_elapsed_sec} | {chunks_sent} | {ack_count} | {partial_count} | {first_partial_sec} | {final_sec} | {final_text_len} | {language} | {send_stall_count} | {max_realtime_lag_sec} | {passed} | `{error}` |".format(
                index=index,
                start_sec=item.get("start_sec"),
                sent_duration_sec=item.get("sent_duration_sec"),
                wall_elapsed_sec=item.get("wall_elapsed_sec"),
                chunks_sent=item.get("chunks_sent"),
                ack_count=item.get("ack_count"),
                partial_count=item.get("partial_count"),
                first_partial_sec=item.get("first_partial_sec"),
                final_sec=item.get("final_sec"),
                final_text_len=item.get("final_text_len"),
                language=item.get("language"),
                send_stall_count=item.get("send_stall_count"),
                max_realtime_lag_sec=item.get("max_realtime_lag_sec"),
                passed="yes" if item.get("passed") else "no",
                error=str(error).replace("`", "'"),
            )
        )
    lines.append("")
    lines.append("## Text Excerpts")
    for index, item in enumerate(result.get("window_summaries", []), start=1):
        lines.append(f"### Window {index}")
        lines.append(f"- Passed: `{'yes' if item.get('passed') else 'no'}`")
        lines.append(f"- Excerpt: `{item.get('final_excerpt') or ''}`")
    lines.append("")
    lines.append("## Conclusion")
    if result.get("passed"):
        lines.append("- Windowed realtime strategy completed the long audio by using independent WebSocket sessions per window.")
    else:
        lines.append("- Windowed realtime strategy did not complete every planned window; inspect the first failed window error above.")
    lines.append("- This validates a service-layer rolling-window workaround, not KV-cache true incremental streaming.")
    lines.append("- The native model path still relies on cumulative per-session generate behavior inside each window.")
    lines.append("")
    lines.append("## Aggregate Text Excerpt")
    lines.append(f"`{_short_text(result.get('aggregate_text', ''), 1000)}`")

    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_windowed(args: argparse.Namespace) -> Dict[str, Any]:
    source_wav = load_audio(args.input)
    source_audio_duration_sec = len(source_wav) / WAV_SAMPLE_RATE
    windows = _window_plan(source_audio_duration_sec, args.window_sec, args.overlap_sec)
    if args.max_windows is not None:
        windows = windows[: args.max_windows]

    case_dir = Path(args.case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    cases: List[Dict[str, Any]] = []
    stopped_early = False
    started_at = _utc_now()
    t0 = time.perf_counter()
    gpu_before = _gpu_snapshot()

    for window in windows:
        window_index = int(window["window_index"])
        case_label = f"{args.case_label}_window_{window_index:03d}"
        case_output = case_dir / f"{case_label}.json"
        event_jsonl = case_dir / f"{case_label}_events.jsonl"
        print(
            f"[Window {window_index}/{len(windows)}] start={window['start_sec']}s duration={window['duration_sec']}s",
            flush=True,
        )
        case = await run_case(
            uri=args.uri,
            audio_path=args.input,
            reference_path=args.reference,
            output_path=str(case_output),
            event_jsonl=str(event_jsonl) if args.write_event_jsonl else None,
            case_label=case_label,
            start_sec=float(window["start_sec"]),
            duration_sec=float(window["duration_sec"]),
            audio_data=source_wav,
            chunk_ms=args.chunk_ms,
            chunk_size_sec=args.chunk_size_sec,
            unfixed_chunk_num=args.unfixed_chunk_num,
            unfixed_token_num=args.unfixed_token_num,
            realtime=True,
            speedup=1.0,
            receive_timeout_sec=args.receive_timeout_sec,
            send_timeout_sec=args.send_timeout_sec,
            ack_timeout_sec=args.ack_timeout_sec,
            max_inflight_chunks=args.max_inflight_chunks,
            context=args.context,
            language=args.language,
        )
        case_output.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
        cases.append(case)
        print(
            f"[Window {window_index}] passed={case.get('passed')} wall={case.get('wall_elapsed_sec')}s text_len={(case.get('final') or {}).get('text_len')}",
            flush=True,
        )
        if args.stop_on_failure and not case.get("passed"):
            stopped_early = True
            break
        if args.inter_window_sleep_sec > 0:
            await asyncio.sleep(args.inter_window_sleep_sec)

    gpu_after = _gpu_snapshot()
    result = _build_result(
        args=args,
        source_audio_duration_sec=source_audio_duration_sec,
        windows=windows,
        cases=cases,
        started_at=started_at,
        wall_elapsed_sec=time.perf_counter() - t0,
        stopped_early=stopped_early,
        gpu_before=gpu_before,
        gpu_after=gpu_after,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.report:
        _write_markdown_report(result, args.report)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run long audio through independent realtime native WebSocket windows."
    )
    parser.add_argument("--uri", default="ws://127.0.0.1:10012/ws/stream")
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--case-label", default="sample_2_windowed")
    parser.add_argument("--window-sec", type=float, default=120.0)
    parser.add_argument("--overlap-sec", type=float, default=0.0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--chunk-ms", type=int, default=500)
    parser.add_argument("--chunk-size-sec", type=float, default=1.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--max-inflight-chunks", type=int, default=4)
    parser.add_argument("--send-timeout-sec", type=float, default=30.0)
    parser.add_argument("--ack-timeout-sec", type=float, default=120.0)
    parser.add_argument("--receive-timeout-sec", type=float, default=300.0)
    parser.add_argument("--inter-window-sleep-sec", type=float, default=0.0)
    parser.add_argument("--context", default="")
    parser.add_argument("--language", default=None)
    parser.add_argument("--text-separator", default="\n")
    parser.add_argument("--write-event-jsonl", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true", default=True)
    parser.add_argument("--continue-on-failure", dest="stop_on_failure", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = asyncio.run(run_windowed(args))
    except Exception as exc:
        result = {
            "schema_version": 1,
            "created_at": _utc_now(),
            "case_label": getattr(args, "case_label", None),
            "passed": False,
            "client_error": {"event": "error", "message": str(exc), "type": type(exc).__name__},
            "config": vars(args),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.report:
            _write_markdown_report({**result, "totals": {}, "window_summaries": []}, args.report)

    print(json.dumps({
        "output": args.output,
        "report": args.report,
        "passed": result.get("passed"),
        "window_count_completed": result.get("window_count_completed"),
        "wall_elapsed_sec": result.get("wall_elapsed_sec"),
        "totals": result.get("totals"),
        "client_error": result.get("client_error"),
    }, ensure_ascii=False, indent=2))
    if not result.get("passed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
