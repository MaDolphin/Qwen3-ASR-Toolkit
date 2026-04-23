import argparse
import asyncio
import json

import numpy as np
import websockets

from qwen3_asr_toolkit.audio_tools import WAV_SAMPLE_RATE, load_audio


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime ASR CLI client over WebSocket.")
    parser.add_argument("--input-file", "-i", required=True, help="Input audio/video file path.")
    parser.add_argument(
        "--ws-url",
        "-u",
        default="ws://127.0.0.1:18000/ws/v1/realtime/transcribe",
        help="Realtime websocket URL.",
    )
    parser.add_argument("--context", "-c", default="", help="Prompt context for ASR.")
    parser.add_argument("--chunk-ms", type=int, default=500, help="Chunk size in milliseconds.")
    parser.add_argument("--decode-interval-ms", type=int, default=600, help="Server-side partial decode interval.")
    parser.add_argument("--min-chunk-ms", type=int, default=200, help="Minimum segment size before decoding.")
    parser.add_argument("--finalize-silence-ms", type=int, default=600, help="Silence duration used to finalize a segment.")
    parser.add_argument("--max-segment-sec", type=float, default=20.0, help="Force-finalize a segment after this duration.")
    parser.add_argument(
        "--simulate-realtime",
        action="store_true",
        help="Sleep between chunks to simulate realtime playback.",
    )
    return parser.parse_args()


async def run_realtime_cli(args):
    wav = load_audio(args.input_file).astype(np.float32)
    chunk_samples = max(1, int(WAV_SAMPLE_RATE * (args.chunk_ms / 1000.0)))

    async with websockets.connect(args.ws_url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        if ready.get("event") == "ready":
            print(f"Connected. session_id={ready.get('session_id')}")
            if ready.get("session_defaults"):
                print(f"Server defaults: {ready.get('session_defaults')}")

        await ws.send(
            json.dumps(
                {
                    "event": "start",
                    "context": args.context,
                    "decode_interval_ms": args.decode_interval_ms,
                    "min_chunk_ms": args.min_chunk_ms,
                    "finalize_silence_ms": args.finalize_silence_ms,
                    "max_segment_sec": args.max_segment_sec,
                }
            )
        )
        started = json.loads(await ws.recv())
        if started.get("event") == "started":
            print(f"Streaming started. session={started.get('session', {})}")
        elif started.get("event") == "error":
            raise RuntimeError(started.get("message", "unknown websocket error"))

        for pos in range(0, len(wav), chunk_samples):
            chunk = wav[pos : pos + chunk_samples]
            await ws.send(chunk.astype(np.float32).tobytes())
            message = json.loads(await ws.recv())
            if message.get("event") == "partial":
                if message.get("updated"):
                    print(f"[partial] language={message.get('language', '')} text={message.get('text', '')}")
            elif message.get("event") == "segment_final":
                print(f"[segment_final] language={message.get('language', '')} text={message.get('text', '')}")
            elif message.get("event") == "error":
                raise RuntimeError(message.get("message", "unknown websocket error"))

            if args.simulate_realtime:
                await asyncio.sleep(args.chunk_ms / 1000.0)

        await ws.send(json.dumps({"event": "finish"}))
        final = json.loads(await ws.recv())
        if final.get("event") == "final":
            print("=== Final Result ===")
            print(f"Language: {final.get('language', '')}")
            print(final.get("text", ""))
        else:
            print(final)


def main():
    args = parse_args()
    asyncio.run(run_realtime_cli(args))


if __name__ == "__main__":
    main()
