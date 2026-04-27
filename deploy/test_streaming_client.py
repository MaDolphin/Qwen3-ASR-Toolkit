#!/usr/bin/env python3
"""
Test client for the vLLM FastAPI streaming server (HTTP backend).

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

    print(f"\n{'='*60}")
    print(f"Audio: {audio_path}")
    print(f"Duration: {total_samples / WAV_SAMPLE_RATE:.1f}s")
    print(f"Chunk size: {chunk_ms}ms ({chunk_samples} samples)")
    print(f"WebSocket URI: {uri}")
    print("=" * 60)

    t_start = time.time()
    first_token_time = None
    finish_send_time = None
    token_count = 0
    ack_count = 0
    final_data = None
    tokens = []

    async with websockets.connect(uri) as ws:
        # 1. Send start
        t0 = time.time()
        await ws.send(
            json.dumps(
                {
                    "event": "start",
                    "stream": True,
                    "context": "",
                }
            )
        )
        resp = await ws.recv()
        data = json.loads(resp)
        print(f"[{time.time() - t0:.3f}s] started: {data}")
        if data.get("event") != "started":
            print("Failed to start session.")
            return None

        # 2. Stream audio chunks
        idx = 0
        call_id = 0
        while idx < total_samples:
            end = min(idx + chunk_samples, total_samples)
            chunk = wav[idx:end]
            t_send = time.time()
            await ws.send(chunk.astype(np.float32).tobytes())
            idx = end
            call_id += 1

            # Wait for ack (with timeout)
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(resp)
                if data.get("event") == "ack":
                    ack_count += 1
                    # Only print first and last ack to reduce noise
                    if call_id == 1 or idx >= total_samples:
                        print(
                            f"[{time.time() - t0:.3f}s] ack call={call_id} "
                            f"total_samples={data.get('total_samples')} "
                            f"duration={data.get('duration_sec')}s"
                        )
            except asyncio.TimeoutError:
                print(f"[{time.time() - t0:.3f}s] Warning: no ack for call {call_id}")

        # 3. Send finish
        finish_send_time = time.time()
        await ws.send(json.dumps({"event": "finish"}))
        print(f"[{finish_send_time - t0:.3f}s] finish sent")

        # 4. Receive tokens and final
        try:
            while True:
                resp = await asyncio.wait_for(ws.recv(), timeout=60.0)
                data = json.loads(resp)
                now = time.time()

                if data.get("event") == "token":
                    token_count += 1
                    if first_token_time is None:
                        first_token_time = now
                        print(
                            f"[{now - t0:.3f}s] FIRST TOKEN (latency after finish: "
                            f"{now - finish_send_time:.3f}s)"
                        )
                    tokens.append(data.get("token", ""))

                elif data.get("event") == "final":
                    final_data = data
                    t_end = time.time()
                    print(f"[{t_end - t0:.3f}s] final received")
                    break

                elif data.get("event") == "error":
                    print(f"[{now - t0:.3f}s] ERROR: {data}")
                    return None

        except asyncio.TimeoutError:
            print("Timeout waiting for final result.")
            return None

    # Summary
    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print("Summary")
    print("=" * 60)
    print(f"Total elapsed (wall clock): {t_total:.3f}s")
    print(f"Audio duration: {total_samples / WAV_SAMPLE_RATE:.1f}s")
    print(f"Ack count: {ack_count}")
    print(f"Token count: {token_count}")
    if first_token_time and finish_send_time:
        print(f"Finish-to-first-token latency: {first_token_time - finish_send_time:.3f}s")
    if final_data:
        print(f"Language: {final_data.get('language')}")
        print(f"Text: {final_data.get('text', '')}")
        print(f"Raw: {final_data.get('raw', '')[:100]}...")

    return {
        "audio": audio_path,
        "audio_duration_sec": total_samples / WAV_SAMPLE_RATE,
        "total_elapsed_sec": t_total,
        "ack_count": ack_count,
        "token_count": token_count,
        "first_token_latency_sec": (first_token_time - finish_send_time) if first_token_time and finish_send_time else None,
        "language": final_data.get("language") if final_data else None,
        "text": final_data.get("text", "") if final_data else "",
        "raw": final_data.get("raw", "") if final_data else "",
    }


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

    result = asyncio.run(run_client(args.uri, args.input, args.chunk_ms))
    if result:
        print(f"\nResult JSON:")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
