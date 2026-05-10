import argparse
import json
import sys
from typing import Sequence

import requests

from qwen3_asr_toolkit import offline_cli, realtime_cli

DEFAULT_BASE_URL = "http://127.0.0.1:10012"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-ASR Native 服务命令行客户端。")
    subparsers = parser.add_subparsers(dest="command")
    health = subparsers.add_parser("health", help="检查 Native 服务健康状态。")
    health.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Native 服务 HTTP base URL。")
    health.add_argument("--timeout-sec", type=float, default=30.0, help="HTTP 超时。")
    subparsers.add_parser("offline", help="离线 HTTP 转写；更多参数见 `qwen3-asr-cli offline --help`。")
    subparsers.add_parser("stream", help="在线 WebSocket 转写；更多参数见 `qwen3-asr-cli stream --help`。")
    return parser


def _print_nested_help(command: str) -> None:
    if command == "offline":
        offline_cli.build_parser().print_help()
    elif command == "stream":
        realtime_cli.build_parser().print_help()


def run_health(argv: Sequence[str] | argparse.Namespace) -> None:
    if isinstance(argv, argparse.Namespace):
        args = argv
    else:
        parser = argparse.ArgumentParser(description="检查 Native 服务健康状态。")
        parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Native 服务 HTTP base URL。")
        parser.add_argument("--timeout-sec", type=float, default=30.0, help="HTTP 超时。")
        args = parser.parse_args(list(argv))
    url = args.base_url.rstrip("/") + "/health"
    response = requests.get(url, timeout=args.timeout_sec)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        build_parser().print_help()
        raise SystemExit(2)

    command, rest = raw_args[0], raw_args[1:]
    if command in {"-h", "--help"}:
        build_parser().print_help()
        return
    if command == "offline":
        if rest and rest[0] in {"-h", "--help"}:
            _print_nested_help("offline")
            return
        offline_cli.main(rest)
        return
    if command == "stream":
        if rest and rest[0] in {"-h", "--help"}:
            _print_nested_help("stream")
            return
        realtime_cli.main(rest)
        return
    if command == "health":
        try:
            run_health(rest)
        except requests.RequestException as exc:
            print(f"ERROR: health 请求失败：{exc}", file=sys.stderr)
            raise SystemExit(3) from exc
        return

    print(f"ERROR: unknown command: {command}", file=sys.stderr)
    build_parser().print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
