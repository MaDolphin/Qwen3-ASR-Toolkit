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
  python deploy/vllm_streaming_server_native.py --host 0.0.0.0 --port 10012
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from deploy.streaming_utils import (
    accumulate_buffer,
    build_ack_event,
    build_started_event,
    consume_full_chunks,
    validate_audio_chunk,
)

# FastAPI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Qwen3ASRModel is imported lazily to allow health endpoint even if vLLM is absent
_qwen_asr_model = None
_model_path: Optional[str] = None
_gpu_memory_utilization: float = 0.8
_max_new_tokens: int = 128
_default_chunk_size_sec: float = 1.0
_default_unfixed_chunk_num: int = 2
_default_unfixed_token_num: int = 5
_audio_queue_size: int = 8
_send_queue_size: int = 32
_decode_timeout_sec: float = 0.0


@dataclass
class AudioQueueItem:
    kind: str
    pcm: Optional[np.ndarray] = None
    received_samples: int = 0
    total_samples: int = 0
    received_at: float = 0.0


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
    connection_id = uuid.uuid4().hex[:8]
    print(f"[WS {connection_id}] connected")

    if asr is None:
        await websocket.send_json(
            {"event": "error", "message": "Model not loaded. Check /health."}
        )
        await websocket.close()
        return

    async def decode_call(func, *args):
        task = asyncio.to_thread(func, *args)
        if _decode_timeout_sec > 0:
            return await asyncio.wait_for(task, timeout=_decode_timeout_sec)
        return await task

    async def enqueue_error(send_queue: asyncio.Queue, stop_event: asyncio.Event, message: str):
        stop_event.set()
        try:
            await send_queue.put({"event": "error", "message": message})
        except Exception:
            pass

    state = None
    try:
        first_message = await websocket.receive()
        if "text" not in first_message:
            await websocket.send_json({"event": "error", "message": "Send start before audio."})
            await websocket.close()
            return

        try:
            payload = json.loads(first_message["text"])
        except json.JSONDecodeError:
            await websocket.send_json({"event": "error", "message": "Invalid JSON."})
            await websocket.close()
            return

        if payload.get("event") != "start":
            await websocket.send_json(
                {"event": "error", "message": f"First event must be start, got: {payload.get('event', '')}"}
            )
            await websocket.close()
            return

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
            await websocket.close()
            return

        await websocket.send_json(
            build_started_event(
                stream=True,
                chunk_size_sec=chunk_size_sec,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
            )
        )

        audio_queue: asyncio.Queue[AudioQueueItem] = asyncio.Queue(maxsize=_audio_queue_size)
        send_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_send_queue_size)
        stop_event = asyncio.Event()
        total_received_samples = 0

        async def receiver_task():
            nonlocal total_received_samples
            try:
                while not stop_event.is_set():
                    while audio_queue.full() and not stop_event.is_set():
                        await asyncio.sleep(0.01)
                    if stop_event.is_set():
                        return

                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        stop_event.set()
                        await audio_queue.put(AudioQueueItem(kind="disconnect", received_at=time.perf_counter()))
                        return

                    if "text" in message:
                        try:
                            control = json.loads(message["text"])
                        except json.JSONDecodeError:
                            await enqueue_error(send_queue, stop_event, "Invalid JSON.")
                            return

                        event = control.get("event", "")
                        if event == "finish":
                            await audio_queue.put(AudioQueueItem(kind="finish", received_at=time.perf_counter()))
                            print(
                                f"[WS {connection_id}] finish received audio_q={audio_queue.qsize()} send_q={send_queue.qsize()}"
                            )
                            return
                        if event == "ping":
                            await send_queue.put({"event": "pong"})
                            continue
                        if event == "start":
                            await enqueue_error(send_queue, stop_event, "Session already started.")
                            return

                        await enqueue_error(send_queue, stop_event, f"Unsupported event: {event}")
                        return

                    if "bytes" in message:
                        ok, err_msg, chunk = validate_audio_chunk(message["bytes"])
                        if not ok or chunk is None:
                            await enqueue_error(send_queue, stop_event, err_msg or "Invalid audio chunk.")
                            return

                        total_received_samples += int(chunk.size)
                        await audio_queue.put(
                            AudioQueueItem(
                                kind="audio",
                                pcm=chunk,
                                received_samples=int(chunk.size),
                                total_samples=total_received_samples,
                                received_at=time.perf_counter(),
                            )
                        )
                        await send_queue.put(
                            build_ack_event(
                                received_samples=int(chunk.size),
                                total_samples=total_received_samples,
                            )
                        )
                        continue

                    await enqueue_error(send_queue, stop_event, "Unsupported websocket frame.")
                    return
            except WebSocketDisconnect:
                stop_event.set()
                try:
                    await audio_queue.put(AudioQueueItem(kind="disconnect", received_at=time.perf_counter()))
                except Exception:
                    pass
            except Exception as exc:
                traceback.print_exc()
                await enqueue_error(send_queue, stop_event, f"Receiver error: {exc}")

        async def processor_task():
            try:
                while not stop_event.is_set():
                    item = await audio_queue.get()
                    if item.kind == "disconnect":
                        print(f"[WS {connection_id}] client disconnected")
                        return

                    if item.kind == "finish":
                        t0 = time.perf_counter()
                        await decode_call(asr.finish_streaming_transcribe, state)
                        elapsed = time.perf_counter() - t0
                        print(
                            f"[WS {connection_id}] final decode={elapsed:.3f}s text_len={len(getattr(state, 'text', '') or '')}"
                        )
                        await send_queue.put(
                            {
                                "event": "final",
                                "language": getattr(state, "language", "") or "",
                                "text": getattr(state, "text", "") or "",
                                "chunk_id": getattr(state, "chunk_id", 0),
                            }
                        )
                        return

                    if item.kind != "audio" or item.pcm is None:
                        continue

                    before_chunk_id = int(getattr(state, "chunk_id", 0) or 0)
                    t0 = time.perf_counter()
                    await decode_call(asr.streaming_transcribe, item.pcm, state)
                    elapsed = time.perf_counter() - t0
                    after_chunk_id = int(getattr(state, "chunk_id", 0) or 0)
                    text = getattr(state, "text", "") or ""
                    print(
                        f"[WS {connection_id}] audio samples={item.received_samples} total={item.total_samples} "
                        f"decode={elapsed:.3f}s chunk_id={after_chunk_id} audio_q={audio_queue.qsize()} send_q={send_queue.qsize()} text_len={len(text)}"
                    )
                    if after_chunk_id > before_chunk_id or text.strip():
                        await send_queue.put(
                            {
                                "event": "partial",
                                "language": getattr(state, "language", "") or "",
                                "text": text,
                                "chunk_id": after_chunk_id,
                            }
                        )
            except Exception as exc:
                traceback.print_exc()
                await enqueue_error(send_queue, stop_event, f"Processor error: {exc}")

        async def sender_task():
            try:
                while True:
                    event = await send_queue.get()
                    await websocket.send_json(event)
                    if event.get("event") in {"final", "error"}:
                        stop_event.set()
                        try:
                            await websocket.close()
                        except Exception:
                            pass
                        return
            except WebSocketDisconnect:
                stop_event.set()
            except Exception as exc:
                stop_event.set()
                print(f"[WS {connection_id}] sender stopped: {exc}")

        receiver = asyncio.create_task(receiver_task(), name=f"receiver-{connection_id}")
        processor = asyncio.create_task(processor_task(), name=f"processor-{connection_id}")
        sender = asyncio.create_task(sender_task(), name=f"sender-{connection_id}")
        tasks = {receiver, processor, sender}
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            if sender in done or sender.done():
                break
            if receiver.done() and processor.done():
                stop_event.set()
                sender.cancel()
                break
            for task in done:
                if not task.cancelled():
                    exc = task.exception()
                    if exc is not None:
                        await enqueue_error(send_queue, stop_event, f"Task error: {exc}")
                        break
        stop_event.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        traceback.print_exc()
        try:
            await websocket.send_json(
                {"event": "error", "message": f"Server error: {exc}"}
            )
            await websocket.close()
        except Exception:
            pass
    finally:
        if state is not None and not getattr(state, "_finished", False):
            try:
                asr.finish_streaming_transcribe(state)
            except Exception:
                pass
        print(f"[WS {connection_id}] disconnected")

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
        default=128,
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
    parser.add_argument(
        "--audio-queue-size",
        type=int,
        default=8,
        help="Per-connection queued audio frame limit before websocket backpressure.",
    )
    parser.add_argument(
        "--send-queue-size",
        type=int,
        default=32,
        help="Per-connection queued server event limit.",
    )
    parser.add_argument(
        "--decode-timeout-sec",
        type=float,
        default=0.0,
        help="Per decode timeout in seconds; 0 disables timeout.",
    )
    return parser.parse_args()


def main():
    global _model_path, _gpu_memory_utilization, _max_new_tokens
    global _default_chunk_size_sec, _default_unfixed_chunk_num, _default_unfixed_token_num
    global _audio_queue_size, _send_queue_size, _decode_timeout_sec

    args = parse_args()
    _model_path = args.model_path or os.environ.get("QWEN3_ASR_MODEL_PATH")
    _gpu_memory_utilization = args.gpu_memory_utilization
    _max_new_tokens = args.max_new_tokens
    _default_chunk_size_sec = args.chunk_size_sec
    _default_unfixed_chunk_num = args.unfixed_chunk_num
    _default_unfixed_token_num = args.unfixed_token_num
    _audio_queue_size = max(1, args.audio_queue_size)
    _send_queue_size = max(1, args.send_queue_size)
    _decode_timeout_sec = max(0.0, args.decode_timeout_sec)

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )


if __name__ == "__main__":
    main()
