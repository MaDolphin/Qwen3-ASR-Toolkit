import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import requests

from qwen3_asr_toolkit.cli_utils import build_offline_api_url

DEFAULT_API_URL = "http://127.0.0.1:10012/api/v1/offline/transcribe"


class CliError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="调用 Qwen3-ASR Native 服务进行离线 HTTP 转写。")
    parser.add_argument("--input-file", "-i", required=True, help="输入音频/视频文件路径。")
    parser.add_argument("--server", default="", help="Native 服务 base URL，例如 http://服务器IP:10012。")
    parser.add_argument(
        "--api-url",
        "-u",
        default="",
        help=f"离线转写 API 地址；优先级高于 --server，默认 {DEFAULT_API_URL}。",
    )
    parser.add_argument("--context", "-c", default="", help="ASR 上下文提示。")
    parser.add_argument("--use-forced-aligner", action="store_true", help="请求 forced aligner 时间戳。")
    parser.add_argument("--timeout-sec", type=float, default=1800.0, help="HTTP 请求超时时间。")
    parser.add_argument("--output-json", default=None, help="保存完整 JSON 响应。")
    parser.add_argument("--output-text", default=None, help="保存纯文本转写结果。")
    parser.add_argument("--save-text", action="store_true", help="保存文本到 <input_basename>.txt。")
    parser.add_argument("--print-segments", action="store_true", help="打印 segments 明细。")
    parser.add_argument("--quiet", action="store_true", help="不打印正文，只写输出文件。")
    parser.add_argument(
        "--fail-on-empty",
        dest="fail_on_empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="文本为空时返回非 0 退出码。",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.api_url = args.api_url or build_offline_api_url(args.server)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _write_text(path: str | os.PathLike[str], text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _write_json(path: str | os.PathLike[str], data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def transcribe(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise CliError(f"输入文件不存在：{input_path}", exit_code=2)

    try:
        with input_path.open("rb") as file_obj:
            files = {"audio_file": (input_path.name, file_obj, "application/octet-stream")}
            data = {
                "context": args.context,
                "use_forced_aligner": str(bool(args.use_forced_aligner)).lower(),
            }
            response = requests.post(args.api_url, files=files, data=data, timeout=args.timeout_sec)
            response.raise_for_status()
    except requests.RequestException as exc:
        raise CliError(f"HTTP 请求失败：{exc}", exit_code=3) from exc

    result = response.json()
    if result.get("error"):
        raise CliError(f"服务返回错误：{result.get('error')}", exit_code=1)
    return result


def print_result(args: argparse.Namespace, result: dict[str, Any]) -> None:
    text = result.get("text") or ""
    aligner = result.get("forced_aligner") or {}
    print("=== Offline Transcription ===")
    print(f"File: {args.input_file}")
    print(f"Language: {result.get('language', '')}")
    print(f"Duration: {result.get('audio_duration_sec', '')}s")
    print(f"Segments: {result.get('segment_count', 0)}")
    print(
        "Forced Aligner: "
        f"requested={bool(aligner.get('requested'))} "
        f"available={bool(aligner.get('available'))} "
        f"items={len(aligner.get('items') or [])}"
    )
    if aligner.get("message"):
        print(f"Forced Aligner Message: {aligner.get('message')}")
    if args.print_segments:
        for segment in result.get("segments") or []:
            print(
                "[segment {index}] {start_sec}-{end_sec}s {language}: {text}".format(
                    index=segment.get("index"),
                    start_sec=segment.get("start_sec"),
                    end_sec=segment.get("end_sec"),
                    language=segment.get("language", ""),
                    text=segment.get("text", ""),
                )
            )
    print("\nText:")
    print(text)


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = transcribe(args)
    text = result.get("text") or ""
    if not args.quiet:
        print_result(args, result)

    if args.output_json:
        _write_json(args.output_json, result)
    if args.output_text:
        _write_text(args.output_text, text + "\n")
    if args.save_text:
        save_path = str(Path(args.input_file).with_suffix(".txt"))
        _write_text(save_path, text + "\n")
        print(f"Saved transcript to: {save_path}")

    if args.fail_on_empty and not text.strip():
        raise CliError("转写文本为空。", exit_code=1)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except CliError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()
