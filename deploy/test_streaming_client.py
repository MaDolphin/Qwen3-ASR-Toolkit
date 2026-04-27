#!/usr/bin/env python3
"""
Test client for the vLLM FastAPI streaming server.

Usage:
  python deploy/test_streaming_client.py \
    -i sample/sample_0.mp3 \
    -u ws://127.0.0.1:10012/ws/stream \
    --chunk-ms 500
"""

import argparse
import asyncio
import json
import sys
import time

import numpy as np
import websockets

sys.path.insert(0, "..")
from qwen3_asr_toolkit.audio_tools import load_audio, WAV_SAMPLE_RATE


async def run_client(uri: str, audio_path: str, chunk_ms: int = 500):
    wav = load_audio(audio_path)
    total_samples = len(wav)
    chunk_samples = int(chunk_ms * WAV_SAMPLE_RATE / 1000)

    print(f"Audio: {audio_path}")
    print(f"Duration: {total_samples / WAV_SAMPLE_RATE:.1f}s")
    print(f"Chunk size: {chunk_ms}ms ({chunk_samples} samples)")
    print(f"Connecting to {uri} ...")

    async with websockets.connect(uri) as ws:
        t0 = time.time()

        # 1. Send start
        await ws.send(
            json.dumps(
                {
                    "event": "start",
                    "context": "",
                    "chunk_size_sec": 1.0,
                    "unfixed_chunk_num": 2,
                    "unfixed_token_num": 5,
                }
            )
        )
        resp = await ws.recv()
        data = json.loads(resp)
        print(f"[{time.time() - t0:.3f}s] {data}")
        if data.get("event") != "started":
            print("Failed to start session.")
            return

        # 2. Stream audio chunks
        idx = 0
        call_id = 0
        while idx < total_samples:
            end = min(idx + chunk_samples, total_samples)
            chunk = wav[idx:end]
            await ws.send(chunk.astype(np.float32).tobytes())
            idx = end
            call_id += 1

            # Non-blocking receive (server may send partial after each chunk)
            try:
                while True:
                    resp = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    data = json.loads(resp)
                    if data.get("event") == "partial":
                        print(
                            f"[{time.time() - t0:.3f}s] partial chunk_id={data.get('chunk_id')} "
                            f"lang={data.get('language')!r} text={data.get('text')!r}"
                        )
            except asyncio.TimeoutError:
                pass

        # 3. Send finish
        await ws.send(json.dumps({"event": "finish"}))
        resp = await ws.recv()
        data = json.loads(resp)
        print(f"[{time.time() - t0:.3f}s] {data}")

        elapsed = time.time() - t0
        print(f"\nTotal elapsed: {elapsed:.3f}s")
        print(f"Final text: {data.get('text', '')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input audio file.")
    parser.add_argument(
        "-u",
        "--uri",
        default="ws://127.0.0.1:10012/ws/stream",
        help="WebSocket URI.",
    )
    parser.add_argument("--chunk-ms", type=int, default=500, help="Chunk size in ms.")
    args = parser.parse_args()

    asyncio.run(run_client(args.uri, args.input, args.chunk_ms))


if __name__ == "__main__":
    main()
