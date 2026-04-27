#!/usr/bin/env python3
"""
FastAPI streaming server for Qwen3-ASR (vLLM backend).

Deploy this on the same machine where vLLM serves Qwen3-ASR.
The user manually loads the model via `Qwen3ASRModel.LLM()` on startup.

Endpoints:
  GET  /health       -> Model readiness + connectivity probe
  WS   /ws/stream    -> Real-time streaming transcription

Usage:
  export QWEN3_ASR_MODEL_PATH=/data/ai_work/Qwen3-ASR-1.7B
  python deploy/vllm_streaming_server.py --host 0.0.0.0 --port 10012
"""

import argparse
import asyncio
import json
import os
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# FastAPI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Qwen3ASRModel is imported lazily to allow health endpoint even if vLLM is absent
_qwen_asr_model = None
_model_path: Optional[str] = None
_gpu_memory_utilization: float = 0.8
_max_new_tokens: int = 32
_default_chunk_size_sec: float = 1.0
_default_unfixed_chunk_num: int = 2
_default_unfixed_token_num: int = 5


def _load_model():
    """Load Qwen3ASRModel via vLLM Python API."""
    global _qwen_asr_model
    if _qwen_asr_model is not None:
        return _qwen_asr_model

    from qwen_asr import Qwen3ASRModel

    model_path = _model_path or os.environ.get("QWEN3_ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B")
    print(f"[Model] Loading Qwen3-ASR from: {model_path}")
    _qwen_asr_model = Qwen3ASRModel.LLM(
        model=model_path,
        gpu_memory_utilization=_gpu_memory_utilization,
        max_new_tokens=_max_new_tokens,
    )
    print("[Model] Loaded successfully.")
    return _qwen_asr_model


def _probe_model() -> dict:
    """Probe model readiness without a real inference (lightweight)."""
    if _qwen_asr_model is None:
        return {
            "status": "unavailable",
            "model": _model_path,
            "message": "Model not loaded yet.",
        }

    try:
        # Try a dummy 0.1s silent audio to verify the pipeline works
        sr = 16000
        dummy = np.zeros(int(sr * 0.1), dtype=np.float32)
        state = _qwen_asr_model.init_streaming_state(
            chunk_size_sec=0.1,
            unfixed_chunk_num=0,
            unfixed_token_num=0,
        )
        _qwen_asr_model.streaming_transcribe(dummy, state)
        _qwen_asr_model.finish_streaming_transcribe(state)
        return {
            "status": "ok",
            "model": _model_path,
            "backend": "vllm",
            "message": "Model is ready.",
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "model": _model_path,
            "message": f"Model loaded but probe failed: {exc}",
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: attempt to load model
    try:
        _load_model()
    except Exception as exc:
        print(f"[Warning] Model auto-load failed: {exc}")
        print("[Warning] /health will report unavailable until model is loaded.")
    yield
    # Shutdown
    print("[Server] Shutting down.")


app = FastAPI(
    title="Qwen3-ASR vLLM Streaming Server",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    result = _probe_model()
    status_code = 200 if result["status"] == "ok" else 503
    return JSONResponse(content=result, status_code=status_code)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    asr = _qwen_asr_model

    if asr is None:
        await websocket.send_json(
            {"event": "error", "message": "Model not loaded. Check /health."}
        )
        await websocket.close()
        return

    state = None
    started = False
    chunk_size_sec = _default_chunk_size_sec
    unfixed_chunk_num = _default_unfixed_chunk_num
    unfixed_token_num = _default_unfixed_token_num

    # Internal buffer: client may send arbitrary-length audio frames
    buffer = np.zeros((0,), dtype=np.float32)
    sr = 16000

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

                    chunk_size_sec = float(payload.get("chunk_size_sec", _default_chunk_size_sec))
                    unfixed_chunk_num = int(payload.get("unfixed_chunk_num", _default_unfixed_chunk_num))
                    unfixed_token_num = int(payload.get("unfixed_token_num", _default_unfixed_token_num))
                    context = str(payload.get("context", ""))
                    language = payload.get("language") or None

                    try:
                        state = asr.init_streaming_state(
                            context=context,
                            language=language,
                            chunk_size_sec=chunk_size_sec,
                            unfixed_chunk_num=unfixed_chunk_num,
                            unfixed_token_num=unfixed_token_num,
                        )
                    except Exception as exc:
                        await websocket.send_json(
                            {"event": "error", "message": f"Failed to init streaming state: {exc}"}
                        )
                        continue

                    started = True
                    buffer = np.zeros((0,), dtype=np.float32)
                    await websocket.send_json(
                        {
                            "event": "started",
                            "chunk_size_sec": chunk_size_sec,
                            "unfixed_chunk_num": unfixed_chunk_num,
                            "unfixed_token_num": unfixed_token_num,
                        }
                    )

                elif event == "finish":
                    if not started or state is None:
                        await websocket.send_json(
                            {"event": "error", "message": "Send start before finish."}
                        )
                        continue

                    # Flush any remaining buffered audio
                    if buffer.shape[0] > 0:
                        asr.streaming_transcribe(buffer, state)
                        buffer = np.zeros((0,), dtype=np.float32)

                    asr.finish_streaming_transcribe(state)
                    await websocket.send_json(
                        {
                            "event": "final",
                            "language": getattr(state, "language", "") or "",
                            "text": getattr(state, "text", "") or "",
                            "chunk_id": getattr(state, "chunk_id", 0),
                        }
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
                if not started or state is None:
                    await websocket.send_json(
                        {"event": "error", "message": "Send start before audio."}
                    )
                    continue

                raw = message["bytes"]
                if len(raw) % 4 != 0:
                    await websocket.send_json(
                        {
                            "event": "error",
                            "message": "Audio chunk must be float32 bytes (multiple of 4).",
                        }
                    )
                    continue

                chunk = np.frombuffer(raw, dtype=np.float32).reshape(-1).copy()
                if chunk.size == 0:
                    continue

                # Accumulate into buffer
                buffer = np.concatenate([buffer, chunk], axis=0)
                chunk_size_samples = int(round(chunk_size_sec * sr))

                # Consume full chunks
                while buffer.shape[0] >= chunk_size_samples:
                    feed = buffer[:chunk_size_samples]
                    buffer = buffer[chunk_size_samples:]
                    asr.streaming_transcribe(feed, state)
                    await websocket.send_json(
                        {
                            "event": "partial",
                            "language": getattr(state, "language", "") or "",
                            "text": getattr(state, "text", "") or "",
                            "chunk_id": getattr(state, "chunk_id", 0),
                        }
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
    finally:
        # Clean up: if there is remaining audio and state is valid, finish it
        if started and state is not None and not getattr(state, "_finished", False):
            try:
                if buffer.shape[0] > 0:
                    asr.streaming_transcribe(buffer, state)
                asr.finish_streaming_transcribe(state)
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR vLLM Streaming Server (FastAPI)"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=10012, help="Bind port.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model path or HF repo id. Overrides QWEN3_ASR_MODEL_PATH env.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max new tokens per streaming step.",
    )
    parser.add_argument(
        "--chunk-size-sec",
        type=float,
        default=1.0,
        help="Default audio chunk size in seconds.",
    )
    parser.add_argument(
        "--unfixed-chunk-num",
        type=int,
        default=2,
        help="Default unfixed_chunk_num for streaming state.",
    )
    parser.add_argument(
        "--unfixed-token-num",
        type=int,
        default=5,
        help="Default unfixed_token_num for streaming state.",
    )
    return parser.parse_args()


def main():
    global _model_path, _gpu_memory_utilization, _max_new_tokens
    global _default_chunk_size_sec, _default_unfixed_chunk_num, _default_unfixed_token_num

    args = parse_args()
    _model_path = args.model_path or os.environ.get("QWEN3_ASR_MODEL_PATH")
    _gpu_memory_utilization = args.gpu_memory_utilization
    _max_new_tokens = args.max_new_tokens
    _default_chunk_size_sec = args.chunk_size_sec
    _default_unfixed_chunk_num = args.unfixed_chunk_num
    _default_unfixed_token_num = args.unfixed_token_num

    import uvicorn

    uvicorn.run(
        "deploy.vllm_streaming_server:app",
        host=args.host,
        port=args.port,
        factory=False,
    )


if __name__ == "__main__":
    main()
