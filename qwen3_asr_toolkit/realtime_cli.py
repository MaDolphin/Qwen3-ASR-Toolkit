import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import websockets

from qwen3_asr_toolkit.audio_tools import WAV_SAMPLE_RATE, load_audio
from qwen3_asr_toolkit.cli_utils import build_stream_ws_url

DEFAULT_WS_URL = "ws://127.0.0.1:10012/ws/stream"


class StreamCliError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="调用 Qwen3-ASR Native WebSocket 服务进行在线转写。")
    parser.add_argument("--input-file", "-i", required=True, help="输入音频/视频文件路径。")
    parser.add_argument("--server", default="", help="Native 服务 base URL，例如 http://服务器IP:10012。")
    parser.add_argument(
        "--ws-url",
        "-u",
        default="",
        help=f"Native WebSocket 地址；优先级高于 --server，默认 {DEFAULT_WS_URL}。",
    )
    parser.add_argument("--context", "-c", default="", help="ASR 上下文提示。")
    parser.add_argument("--start-sec", type=float, default=0.0, help="从音频第几秒开始发送。")
    parser.add_argument("--duration-sec", type=float, default=None, help="只发送指定秒数音频。")
    parser.add_argument("--chunk-ms", type=int, default=500, help="客户端发送 chunk 大小。")
    parser.add_argument("--chunk-size-sec", type=float, default=1.0, help="服务端 streaming chunk 参数。")
    parser.add_argument("--unfixed-chunk-num", type=int, default=2, help="Native streaming 参数。")
    parser.add_argument("--unfixed-token-num", type=int, default=5, help="Native streaming 参数。")
    parser.add_argument("--max-inflight-chunks", type=int, default=4, help="最多未 ack chunk 数。")
    parser.add_argument("--send-timeout-sec", type=float, default=30.0, help="单次 send 超时。")
    parser.add_argument("--ack-timeout-sec", type=float, default=120.0, help="等待 ack 超时。")
    parser.add_argument("--receive-timeout-sec", type=float, default=300.0, help="等待服务端事件超时。")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--realtime", action="store_true", default=True, help="按真实时间发送。")
    mode.add_argument("--accelerated", action="store_true", help="不 sleep，调试使用。")
    parser.add_argument("--output-json", default=None, help="保存完整 JSON 指标。")
    parser.add_argument("--event-jsonl", default=None, help="保存逐事件 JSONL。")
    parser.add_argument("--print-partials", dest="print_partials", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true", help="只打印 final 摘要。")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.ws_url = args.ws_url or build_stream_ws_url(args.server)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def build_start_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "event": "start",
        "stream": True,
        "context": args.context,
        "chunk_size_sec": args.chunk_size_sec,
        "unfixed_chunk_num": args.unfixed_chunk_num,
        "unfixed_token_num": args.unfixed_token_num,
    }


def _append_jsonl(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _slice_audio(wav: np.ndarray, start_sec: float, duration_sec: float | None) -> np.ndarray:
    start_sample = max(0, int(start_sec * WAV_SAMPLE_RATE))
    end_sample = len(wav) if duration_sec is None else start_sample + max(0, int(duration_sec * WAV_SAMPLE_RATE))
    return wav[start_sample:end_sample]


async def run_stream(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise StreamCliError(f"输入文件不存在：{input_path}", exit_code=2)

    try:
        full_wav = load_audio(str(input_path)).astype(np.float32)
        wav = _slice_audio(full_wav, args.start_sec, args.duration_sec)
        chunk_samples = max(1, int(WAV_SAMPLE_RATE * args.chunk_ms / 1000.0))
        realtime = not args.accelerated
        chunks_sent = 0
        ack_count = 0
        partial_count = 0
        error_count = 0
        final_payload = None
        partials = []

        if not args.quiet:
            print("=== Native WebSocket Streaming ===")
            print(f"File: {args.input_file}")
            print(f"Mode: {'realtime' if realtime else 'accelerated'}")
            print(f"Chunk: {args.chunk_ms}ms")

        async with websockets.connect(args.ws_url, max_size=None) as ws:
            await ws.send(json.dumps(build_start_payload(args), ensure_ascii=False))
            started = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.receive_timeout_sec))
            _append_jsonl(args.event_jsonl, {"direction": "recv", "event": started.get("event"), "payload": started})
            if started.get("event") == "error":
                raise StreamCliError(started.get("message", "websocket start failed"), exit_code=1)
            if started.get("event") != "started":
                raise StreamCliError(f"未收到 started 事件：{started}", exit_code=3)
            if not args.quiet:
                print(f"Started: sample_rate={started.get('sample_rate')}")

            for pos in range(0, len(wav), chunk_samples):
                chunk = wav[pos : pos + chunk_samples]
                await asyncio.wait_for(ws.send(chunk.astype(np.float32).tobytes()), timeout=args.send_timeout_sec)
                chunks_sent += 1
                _append_jsonl(args.event_jsonl, {"direction": "send", "event": "audio", "chunk_index": chunks_sent})

                while True:
                    message = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.receive_timeout_sec))
                    event = message.get("event")
                    _append_jsonl(args.event_jsonl, {"direction": "recv", "event": event, "payload": message})
                    if event == "ack":
                        ack_count += 1
                        break
                    if event == "partial":
                        partial_count += 1
                        partials.append(message)
                        text = message.get("text") or ""
                        if args.print_partials and not args.quiet and text:
                            print(f"[partial #{partial_count}] {text}")
                    elif event == "error":
                        error_count += 1
                        raise StreamCliError(message.get("message", "server error"), exit_code=1)

                if realtime:
                    await asyncio.sleep(args.chunk_ms / 1000.0)

            await ws.send(json.dumps({"event": "finish"}, ensure_ascii=False))
            _append_jsonl(args.event_jsonl, {"direction": "send", "event": "finish"})
            while True:
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.receive_timeout_sec))
                event = message.get("event")
                _append_jsonl(args.event_jsonl, {"direction": "recv", "event": event, "payload": message})
                if event == "partial":
                    partial_count += 1
                    partials.append(message)
                    continue
                if event == "final":
                    final_payload = message
                    break
                if event == "error":
                    error_count += 1
                    raise StreamCliError(message.get("message", "server error"), exit_code=1)
    except (OSError, websockets.WebSocketException) as exc:
        raise StreamCliError(f"WebSocket 连接失败：{exc}", exit_code=3) from exc
    except asyncio.TimeoutError as exc:
        raise StreamCliError("WebSocket 等待超时。", exit_code=4) from exc

    final_text = (final_payload or {}).get("text") or ""
    result = {
        "passed": bool(final_text) and error_count == 0,
        "config": {
            "ws_url": args.ws_url,
            "input_file": args.input_file,
            "start_sec": args.start_sec,
            "duration_sec": args.duration_sec,
            "chunk_ms": args.chunk_ms,
            "mode": "realtime" if not args.accelerated else "accelerated",
        },
        "counts": {
            "chunks_sent": chunks_sent,
            "ack_count": ack_count,
            "partial_count": partial_count,
            "error_count": error_count,
        },
        "final": final_payload or {},
        "partials_tail": partials[-5:],
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Final Result ===")
    print(f"Language: {(final_payload or {}).get('language', '')}")
    print(f"Chunks sent: {chunks_sent}")
    print(f"Ack count: {ack_count}")
    print(f"Partial count: {partial_count}")
    print("\nText:")
    print(final_text)

    if not result["passed"]:
        raise StreamCliError("WebSocket 转写未通过：final 文本为空或存在错误。", exit_code=1)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(run_stream(args))
    except StreamCliError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()
