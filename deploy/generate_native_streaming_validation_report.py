#!/usr/bin/env python3
"""Generate a Markdown report from native streaming validation artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_command(args: List[str]) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return {
            "command": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"command": args, "error": str(exc)}


def _short_text(text: str, limit: int = 300) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _strip_reference_language(text: str) -> str:
    first_line, _, rest = text.partition("\n")
    if first_line.strip().lower().startswith("language:"):
        return rest.strip()
    return text.strip()


def _read_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return {"missing": True, "path": str(target)}
    return json.loads(target.read_text(encoding="utf-8"))


def _sample_summary(case: Dict[str, Any]) -> Dict[str, Any]:
    config = case.get("config", {})
    final = case.get("final", {})
    latency = case.get("latency", {})
    counts = case.get("event_counts", {})
    score = case.get("score") or {}
    reference_path = config.get("reference")
    reference_text = ""
    if reference_path and Path(reference_path).exists():
        reference_text = _strip_reference_language(
            Path(reference_path).read_text(encoding="utf-8")
        )
    return {
        "audio": config.get("audio", ""),
        "reference": reference_path,
        "mode": config.get("mode", ""),
        "chunk_ms": config.get("chunk_ms"),
        "audio_duration_sec": case.get("audio_duration_sec"),
        "ack_count": counts.get("ack"),
        "partial_count": counts.get("partial"),
        "first_partial_sec": latency.get("first_partial_sec"),
        "first_nonempty_partial_sec": latency.get("first_nonempty_partial_sec"),
        "final_sec": latency.get("final_sec"),
        "final_text_len": final.get("text_len"),
        "language": final.get("language", ""),
        "passed": case.get("passed"),
        "wer": score.get("wer"),
        "cer": score.get("cer"),
        "final_excerpt": _short_text(final.get("text", ""), 300),
        "reference_excerpt": _short_text(reference_text, 300),
        "error": (case.get("error") or {}).get("message"),
    }


def _parse_gpu_csv(snapshot: Optional[Dict[str, Any]]) -> str:
    if not snapshot:
        return "N/A"
    stdout = snapshot.get("stdout") or ""
    if stdout:
        return stdout
    if snapshot.get("error"):
        return f"ERROR: {snapshot['error']}"
    stderr = snapshot.get("stderr")
    if stderr:
        return f"STDERR: {stderr}"
    return "N/A"


def _key_package_versions() -> str:
    from importlib import metadata

    names = [
        "numpy",
        "torch",
        "torchaudio",
        "vllm",
        "transformers",
        "fastapi",
        "uvicorn",
        "websockets",
        "librosa",
        "soundfile",
        "av",
        "qwen3-asr-toolkit",
    ]
    lines = []
    for name in names:
        try:
            lines.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{name}=not-installed")
    return "\n".join(lines)


def _tail_lines(path: Optional[str], limit: int = 60) -> str:
    if not path:
        return ""
    target = Path(path)
    if not target.exists():
        return ""
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def build_report(
    *,
    output: Path,
    cases: List[Dict[str, Any]],
    health_json: Optional[Dict[str, Any]],
    health_status: Optional[str],
    health_body_path: Optional[str],
    env_name: str,
    model_path: str,
    server_log_path: Optional[str],
    gpu_before: Optional[Dict[str, Any]],
    gpu_after_load: Optional[Dict[str, Any]],
    gpu_after_all: Optional[Dict[str, Any]],
    python_info: Dict[str, Any],
    package_info: Dict[str, Any],
    config: Dict[str, Any],
    notes: Optional[str],
) -> str:
    case_summaries = [_sample_summary(case) for case in cases]
    all_passed = bool(case_summaries) and all(item["passed"] for item in case_summaries)
    partial_supported = all(
        (item.get("partial_count") or 0) > 0 for item in case_summaries
    ) if case_summaries else False
    long_case = next(
        (item for item in case_summaries if item.get("audio", "").endswith("sample_2.m4a")),
        None,
    )

    lines: List[str] = []
    lines.append("# Native Streaming Validation Report")
    lines.append("")
    lines.append(f"- Generated at: `{_utc_now()}`")
    lines.append(f"- Conda environment: `{env_name}`")
    lines.append(f"- Python: `{python_info.get('stdout') or python_info.get('error') or 'N/A'}`")
    lines.append(f"- Model path: `{model_path}`")
    lines.append("")
    lines.append("## 1. 测试环境")
    lines.append(f"- Workspace: `{PROJECT_ROOT}`")
    lines.append(f"- Python executable: `{sys.executable}`")
    lines.append(f"- GPU before load: `{_parse_gpu_csv(gpu_before)}`")
    lines.append(f"- GPU after load: `{_parse_gpu_csv(gpu_after_load)}`")
    lines.append(f"- GPU after all tests: `{_parse_gpu_csv(gpu_after_all)}`")
    lines.append(f"- Key packages: `{package_info.get('stdout') or package_info.get('error') or 'N/A'}`")
    lines.append("")
    lines.append("## 2. 服务配置")
    lines.append(f"- Host/port: `{config.get('host')}:{config.get('port')}`")
    lines.append(f"- `gpu_memory_utilization`: `{config.get('gpu_memory_utilization')}`")
    lines.append(f"- `max_new_tokens`: `{config.get('max_new_tokens')}`")
    lines.append(f"- `chunk_size_sec`: `{config.get('chunk_size_sec')}`")
    lines.append(f"- `unfixed_chunk_num`: `{config.get('unfixed_chunk_num')}`")
    lines.append(f"- `unfixed_token_num`: `{config.get('unfixed_token_num')}`")
    lines.append("")
    lines.append("## 3. 健康检查结果")
    lines.append(f"- HTTP status: `{health_status or 'N/A'}`")
    if health_json is not None:
        lines.append("- Body:")
        lines.append("```json")
        lines.append(json.dumps(health_json, ensure_ascii=False, indent=2))
        lines.append("```")
    elif health_body_path and Path(health_body_path).exists():
        lines.append(f"- Raw body path: `{health_body_path}`")
    else:
        lines.append("- Body: `N/A`")
    lines.append("")
    lines.append("## 4. 样本结果表")
    lines.append("| 样本 | 模式 | 时长(s) | chunk-ms | ack | partial | 首 partial(s) | 首非空 partial(s) | final(s) | final 长度 | language | WER | CER | 通过 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|")
    for item in case_summaries:
        lines.append(
            "| `{audio}` | {mode} | {audio_duration_sec} | {chunk_ms} | {ack_count} | {partial_count} | {first_partial_sec} | {first_nonempty_partial_sec} | {final_sec} | {final_text_len} | {language} | {wer} | {cer} | {passed} |".format(
                audio=item["audio"],
                mode=item["mode"],
                audio_duration_sec=item["audio_duration_sec"],
                chunk_ms=item["chunk_ms"],
                ack_count=item["ack_count"],
                partial_count=item["partial_count"],
                first_partial_sec=item["first_partial_sec"],
                first_nonempty_partial_sec=item["first_nonempty_partial_sec"],
                final_sec=item["final_sec"],
                final_text_len=item["final_text_len"],
                language=item["language"],
                wer=item["wer"],
                cer=item["cer"],
                passed="yes" if item["passed"] else "no",
            )
        )
    lines.append("")
    lines.append("## 5. 文本输出摘要")
    for item in case_summaries:
        lines.append(f"### `{item['audio']}`")
        lines.append(f"- 通过: `{'yes' if item['passed'] else 'no'}`")
        if item.get("error"):
            lines.append(f"- Error: `{item['error']}`")
        lines.append(f"- Final excerpt: `{item['final_excerpt']}`")
        lines.append(f"- Reference excerpt: `{item['reference_excerpt']}`")
    lines.append("")
    lines.append("## 6. 关键结论")
    lines.append(
        f"- Native WebSocket streaming 服务闭环: `{'通过' if all_passed else '未通过'}`"
    )
    lines.append(
        f"- 真实数据 partial 输出: `{'已观察到' if partial_supported else '未稳定观察到'}`"
    )
    if long_case:
        lines.append(
            f"- 长音频 `sample_2.m4a` final 延迟: `{long_case.get('final_sec')}` 秒，partial 数量 `{long_case.get('partial_count')}`"
        )
    lines.append(
        "- True streaming 结论: `当前底层实现仍是累计音频重送 generate，不应表述为已确认 KV-cache 真增量 streaming`"
    )
    lines.append("")
    lines.append("## 7. 已知限制")
    lines.append(
        "- `qwen_asr/inference/qwen3_asr.py` 当前 `streaming_transcribe()` 会把累计音频重新送入 `self.model.generate(...)`。"
    )
    lines.append("- Native server 只提供 `partial`/`final`，没有 `segment_final`。")
    lines.append("- 当前验证关注服务层功能闭环与时延，不证明底层 KV cache 增量推理已生效。")
    lines.append("- 无 forced aligner 时间戳。")
    lines.append("")
    lines.append("## 8. 后续建议")
    lines.append("- 若需真正 KV-cache 增量 streaming，需要修改底层推理实现，而非仅调整服务端协议。")
    lines.append("- 若 Python 3.13 与 `vllm==0.14.0` 不兼容，建议切到 Python 3.11/3.12 复测。")
    if notes:
        lines.append(f"- 附加备注: `{notes}`")
    if server_log_path:
        lines.append("")
        lines.append("## Server Log Tail")
        lines.append("```text")
        lines.append(_tail_lines(server_log_path) or "(empty)")
        lines.append("```")

    report = "\n".join(lines) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate native streaming validation Markdown report.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--health-json", default=None)
    parser.add_argument("--health-status", default=None)
    parser.add_argument("--health-body", default=None)
    parser.add_argument("--server-log", default=None)
    parser.add_argument("--gpu-before", default=None)
    parser.add_argument("--gpu-after-load", default=None)
    parser.add_argument("--gpu-after-all", default=None)
    parser.add_argument("--env-name", default=os.environ.get("CONDA_DEFAULT_ENV", "unknown"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10012)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--chunk-size-sec", type=float, default=1.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--notes", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    health_json = _read_json(args.health_json)
    gpu_before = _read_json(args.gpu_before)
    gpu_after_load = _read_json(args.gpu_after_load)
    gpu_after_all = _read_json(args.gpu_after_all)
    cases = [_read_json(path) for path in args.case]
    cases = [case for case in cases if case is not None]
    python_info = _run_command([sys.executable, "-V"])
    package_info = {"stdout": _key_package_versions()}
    report = build_report(
        output=Path(args.output),
        cases=cases,
        health_json=health_json,
        health_status=args.health_status,
        health_body_path=args.health_body,
        env_name=args.env_name,
        model_path=args.model_path,
        server_log_path=args.server_log,
        gpu_before=gpu_before,
        gpu_after_load=gpu_after_load,
        gpu_after_all=gpu_after_all,
        python_info=python_info,
        package_info=package_info,
        config={
            "host": args.host,
            "port": args.port,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_new_tokens": args.max_new_tokens,
            "chunk_size_sec": args.chunk_size_sec,
            "unfixed_chunk_num": args.unfixed_chunk_num,
            "unfixed_token_num": args.unfixed_token_num,
        },
        notes=args.notes,
    )
    print(report)


if __name__ == "__main__":
    main()
