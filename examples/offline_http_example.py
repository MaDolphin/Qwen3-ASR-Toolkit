from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import requests

from client.cli.url_utils import build_offline_api_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-ASR 离线 HTTP Python 调用示例。")
    parser.add_argument("--server", default="http://127.0.0.1:10012")
    parser.add_argument("--input", default="sample/sample_0.mp3")
    parser.add_argument("--context", default="")
    parser.add_argument("--use-forced-aligner", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    api_url = build_offline_api_url(args.server)
    with input_path.open("rb") as file_obj:
        response = requests.post(
            api_url,
            files={"audio_file": (input_path.name, file_obj, "application/octet-stream")},
            data={
                "context": args.context,
                "use_forced_aligner": str(bool(args.use_forced_aligner)).lower(),
            },
            timeout=args.timeout_sec,
        )
    response.raise_for_status()
    result = response.json()
    print(f"language={result.get('language')} text_len={len(result.get('text') or '')} segments={result.get('segment_count')}")
    print((result.get("text") or "")[:300])
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
