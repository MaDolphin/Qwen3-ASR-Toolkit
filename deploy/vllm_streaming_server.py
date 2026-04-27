#!/usr/bin/env python3
"""
FastAPI streaming server for Qwen3-ASR (vLLM HTTP backend).

This server does NOT load the model locally. Instead, it calls the vLLM
OpenAI-compatible HTTP API (including /v1/audio/transcriptions with stream=true).

Deploy this on any machine that can reach the vLLM service.
When co-located with vLLM, bind to 0.0.0.0.

Endpoints:
  GET  /health       -> vLLM backend connectivity probe
  WS   /ws/stream    -> Streaming transcription (accumulates audio then forwards
                        to vLLM /v1/audio/transcriptions, supports SSE streaming)

Usage:
  export VLLM_BASE_URL=http://127.0.0.1:10010/v1
  export VLLM_API_KEY=EMPTY
  export QWEN3_ASR_MODEL=Qwen3-ASR-1.7B
  python deploy/vllm_streaming_server.py --host 0.0.0.0 --port 10012
"""

import argparse
import asyncio
import io
import json
import os
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf

# FastAPI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Async HTTP client
import httpx

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qwen_asr.inference.utils import parse_asr_output
from deploy.streaming_utils import (
    accumulate_buffer,
    build_ack_event,
    build_started_event,
    clean_duplicate_asr_prefixes,
    validate_audio_chunk,
)

# Config globals
_vllm_base_url: Optional[str] = None
_vllm_api_key: Optional[str] = None
_model_name: Optional[str] = None
_default_chunk_size_sec: float = 1.0


def _get_vllm_url(path: str = "") -> str:
    base = _vllm_base_url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _get_api_key() -> str:
    return _vllm_api_key or os.environ.get("VLLM_API_KEY", "EMPTY")


def _get_model() -> str:
    return _model_name or os.environ.get("QWEN3_ASR_MODEL", "Qwen3-ASR-1.7B")


async def _probe_vllm() -> dict:
    """Probe vLLM backend readiness via /models endpoint."""
    url = _get_vllm_url("models")
    headers = {"Authorization": f"Bearer {_get_api_key()}"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("id", "") for m in data.get("data", [])]
            target = _get_model()
            if target in models:
                return {
                    "status": "ok",
                    "backend": "vllm-http",
                    "model": target,
                    "available_models": models,
                    "message": "vLLM backend is reachable and model is available.",
                }
            return {
                "status": "degraded",
                "backend": "vllm-http",
                "model": target,
                "available_models": models,
                "message": f"vLLM reachable but model '{target}' not found in available models.",
            }
        return {
            "status": "unavailable",
            "backend": "vllm-http",
            "model": _get_model(),
            "message": f"vLLM returned HTTP {resp.status_code}: {resp.text[:200]}",
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "backend": "vllm-http",
            "model": _get_model(),
            "message": f"Cannot reach vLLM backend: {exc}",
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[Config] vLLM base URL: {_get_vllm_url()}")
    print(f"[Config] Model: {_get_model()}")
    yield
    print("[Server] Shutting down.")


app = FastAPI(
    title="Qwen3-ASR vLLM Streaming Server (HTTP Backend)",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    result = await _probe_vllm()
    status_code = 200 if result["status"] == "ok" else 503
    return JSONResponse(content=result, status_code=status_code)


async def _call_vllm_transcriptions(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    stream: bool = True,
):
    """Call vLLM /v1/audio/transcriptions via HTTP."""
    url = _get_vllm_url("audio/transcriptions")
    headers = {"Authorization": f"Bearer {_get_api_key()}"}

    async with httpx.AsyncClient(timeout=300.0) as client:
        files = {"file": (filename, io.BytesIO(audio_bytes), "audio/wav")}
        data = {"model": _get_model(), "stream": "true" if stream else "false"}
        resp = await client.post(url, headers=headers, files=files, data=data)

    if resp.status_code != 200:
        raise RuntimeError(
            f"vLLM transcription failed ({resp.status_code}): {resp.text[:500]}"
        )

    return resp


async def _parse_sse_stream(resp: httpx.Response):
    """Parse SSE stream from vLLM /v1/audio/transcriptions?stream=true."""
    async for line in resp.aiter_lines():
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content", "")
        finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
        yield content, finish_reason


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    started = False
    buffer = np.zeros((0,), dtype=np.float32)
    sr = 16000
    stream_mode = True
    context = ""
    language = None

    try:
        while True:
            message = await websocket.receive()

            # Text control messages (JSON)
            if "text" in message:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"event": "error", "message": "Invalid JSON."}
                    )
                    continue

                event = payload.get("event", "")

                if event == "start":
                    if started:
                        await websocket.send_json(
                            {"event": "error", "message": "Session already started."}
                        )
                        continue

                    stream_mode = bool(payload.get("stream", True))
                    context = str(payload.get("context", ""))
                    language = payload.get("language") or None
                    started = True
                    buffer = np.zeros((0,), dtype=np.float32)
                    await websocket.send_json(build_started_event(stream=stream_mode))

                elif event == "finish":
                    if not started:
                        await websocket.send_json(
                            {"event": "error", "message": "Send start before finish."}
                        )
                        continue

                    # Convert accumulated float32 PCM to WAV bytes
                    if buffer.shape[0] == 0:
                        await websocket.send_json(
                            {"event": "error", "message": "No audio received."}
                        )
                        continue

                    wav_io = io.BytesIO()
                    sf.write(wav_io, buffer, sr, format="WAV")
                    audio_bytes = wav_io.getvalue()

                    try:
                        # Call vLLM transcription
                        resp = await _call_vllm_transcriptions(
                            audio_bytes,
                            filename="audio.wav",
                            stream=stream_mode,
                        )

                        if stream_mode:
                            # Stream SSE chunks back to WebSocket client
                            full_raw = ""
                            async for token, finish_reason in _parse_sse_stream(resp):
                                prev_raw = full_raw
                                next_raw = full_raw + token
                                # Clean duplicate language<asr_text> prefixes that vLLM
                                # intermittently re-emits during long-audio streaming.
                                full_raw = clean_duplicate_asr_prefixes(next_raw)
                                clean_token = (
                                    full_raw[len(prev_raw) :]
                                    if full_raw.startswith(prev_raw)
                                    else token
                                )
                                await websocket.send_json(
                                    {
                                        "event": "token",
                                        "token": clean_token,
                                        "text_so_far": full_raw,
                                    }
                                )
                            # Parse final result
                            lang, text = parse_asr_output(full_raw, user_language=language)
                            await websocket.send_json(
                                {
                                    "event": "final",
                                    "language": lang,
                                    "text": text,
                                    "raw": full_raw,
                                }
                            )
                        else:
                            # Non-streaming: return complete result
                            result = resp.json()
                            raw_text = result.get("text", "")
                            lang, text = parse_asr_output(raw_text, user_language=language)
                            await websocket.send_json(
                                {
                                    "event": "final",
                                    "language": lang,
                                    "text": text,
                                    "raw": raw_text,
                                }
                            )

                    except Exception as exc:
                        traceback.print_exc()
                        await websocket.send_json(
                            {"event": "error", "message": f"Transcription failed: {exc}"}
                        )

                    await websocket.close()
                    return

                elif event == "ping":
                    await websocket.send_json({"event": "pong"})

                else:
                    await websocket.send_json(
                        {"event": "error", "message": f"Unsupported event: {event}"}
                    )

            # Binary audio frames
            elif "bytes" in message:
                if not started:
                    await websocket.send_json(
                        {"event": "error", "message": "Send start before audio."}
                    )
                    continue

                ok, err_msg, chunk = validate_audio_chunk(message["bytes"])
                if not ok:
                    await websocket.send_json({"event": "error", "message": err_msg})
                    continue

                buffer = accumulate_buffer(buffer, chunk)
                # In HTTP-backend mode, we only accumulate; no partial results until finish.
                # Send back an ack so the client knows the chunk was received.
                await websocket.send_json(
                    build_ack_event(received_samples=chunk.size, total_samples=buffer.size)
                )

            else:
                await websocket.send_json(
                    {"event": "error", "message": "Unsupported websocket frame."}
                )

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        traceback.print_exc()
        try:
            await websocket.send_json(
                {"event": "error", "message": f"Server error: {exc}"}
            )
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR vLLM Streaming Server (HTTP Backend)"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=10012, help="Bind port.")
    parser.add_argument(
        "--vllm-base-url",
        type=str,
        default=None,
        help="vLLM OpenAI-compatible base URL. Overrides VLLM_BASE_URL env.",
    )
    parser.add_argument(
        "--vllm-api-key",
        type=str,
        default=None,
        help="vLLM API key. Overrides VLLM_API_KEY env.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="ASR model name served by vLLM. Overrides QWEN3_ASR_MODEL env.",
    )
    return parser.parse_args()


def main():
    global _vllm_base_url, _vllm_api_key, _model_name

    args = parse_args()
    _vllm_base_url = args.vllm_base_url
    _vllm_api_key = args.vllm_api_key
    _model_name = args.model

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
