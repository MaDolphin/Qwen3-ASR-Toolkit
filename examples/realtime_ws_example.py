from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from client.cli.stream import parse_args, run_stream
from client.cli.url_utils import build_stream_ws_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-ASR WebSocket Python 调用示例。")
    parser.add_argument("--server", default="http://127.0.0.1:10012")
    parser.add_argument("--input", default="sample/sample_2.m4a")
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--chunk-ms", type=int, default=500)
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--accelerated", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stream_args = parse_args([
        "--ws-url",
        build_stream_ws_url(args.server),
        "--input-file",
        args.input,
        "--duration-sec",
        str(args.duration_sec),
        "--chunk-ms",
        str(args.chunk_ms),
        "--output-json",
        args.output_json or "runtime/examples/ws_result.json",
        "--quiet",
    ] + (["--accelerated"] if args.accelerated else ["--realtime"]))
    result = asyncio.run(run_stream(stream_args))
    print(json.dumps({"passed": result.get("passed"), "counts": result.get("counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
