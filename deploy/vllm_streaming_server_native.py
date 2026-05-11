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
import concurrent.futures
import json
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from qwen3_asr_toolkit.offline_transcriber import OfflineTranscriber

# Qwen3ASRModel is imported lazily to allow health endpoint even if vLLM is absent
_qwen_asr_model = None
_model_path: Optional[str] = None
_gpu_memory_utilization: float = 0.5
_max_new_tokens: int = 128
_kv_cache_memory_bytes: Optional[int] = None
_cpu_offload_gb: float = 0.0
_dtype: str = "auto"
_max_model_len: int = 65536
_enforce_eager: bool = False
_default_chunk_size_sec: float = 1.0
_default_unfixed_chunk_num: int = 2
_default_unfixed_token_num: int = 5
_audio_queue_size: int = 8
_send_queue_size: int = 32
_decode_timeout_sec: float = 0.0
_enable_offline_api: bool = True
_offline_num_threads: int = 1
_vad_target_segment_s: int = 45
_vad_max_segment_s: int = 60
_aligner_mode: str = "disabled"
_aligner_model_path: Optional[str] = None
_aligner_base_url: str = ""
_aligner_api_key: str = "EMPTY"
_aligner_timeout_s: int = 120
_aligner_timestamp_segment_time_ms: int = 80
_max_concurrent_asr_jobs: int = 1
_requested_max_concurrent_asr_jobs: int = 1
_offline_transcriber: Optional[OfflineTranscriber] = None
_asr_scheduler: Optional["ASRScheduler"] = None


@dataclass
class AudioQueueItem:
    kind: str
    pcm: Optional[np.ndarray] = None
    received_samples: int = 0
    total_samples: int = 0
    received_at: float = 0.0


@dataclass(order=True)
class ASRJob:
    priority: int
    sequence: int
    fn: Callable[[], Any] = field(compare=False)
    future: concurrent.futures.Future = field(compare=False)
    label: str = field(default="", compare=False)
    submitted_at: float = field(default_factory=time.perf_counter, compare=False)


class ASRScheduler:
    """Single-worker ASR scheduler with realtime-priority queue."""

    PRIORITY_REALTIME = 0
    PRIORITY_OFFLINE = 10

    def __init__(self):
        self._queue: queue.PriorityQueue[ASRJob] = queue.PriorityQueue()
        self._sequence = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, name="asr-priority-worker", daemon=True)
        self._worker.start()

    def submit(self, fn: Callable[[], Any], *, priority: int, label: str = ""):
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        self._queue.put(ASRJob(priority=priority, sequence=sequence, fn=fn, future=future, label=label))
        return future

    def run_sync(self, fn: Callable[[], Any], *, priority: int, label: str = ""):
        return self.submit(fn, priority=priority, label=label).result()

    async def run_async(self, fn: Callable[[], Any], *, priority: int, label: str = "", timeout: float = 0.0):
        future = self.submit(fn, priority=priority, label=label)
        wrapped = asyncio.wrap_future(future)
        if timeout and timeout > 0:
            return await asyncio.wait_for(wrapped, timeout=timeout)
        return await wrapped

    def qsize(self) -> int:
        return self._queue.qsize()

    def shutdown(self) -> None:
        self._stop_event.set()
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put(ASRJob(priority=999999, sequence=999999999, fn=lambda: None, future=future, label="shutdown"))
        self._worker.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            job = self._queue.get()
            if job.label == "shutdown":
                return
            if job.future.cancelled():
                continue
            try:
                wait_sec = time.perf_counter() - job.submitted_at
                if wait_sec > 0.1:
                    print(f"[ASR Scheduler] start label={job.label} priority={job.priority} wait={wait_sec:.3f}s queued={self.qsize()}")
                result = job.fn()
            except Exception as exc:
                job.future.set_exception(exc)
            else:
                job.future.set_result(result)


def _scheduler_required() -> ASRScheduler:
    if _asr_scheduler is None:
        raise RuntimeError("ASR scheduler is not initialized.")
    return _asr_scheduler



def _parse_size_bytes(value: str | int | None) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    raw = str(value).strip()
    if raw == "" or raw == "0":
        return None
    lower = raw.lower().replace(" ", "")
    units = [
        ("gib", 1024 ** 3),
        ("gb", 1024 ** 3),
        ("g", 1024 ** 3),
        ("mib", 1024 ** 2),
        ("mb", 1024 ** 2),
        ("m", 1024 ** 2),
        ("kib", 1024),
        ("kb", 1024),
        ("k", 1024),
    ]
    for suffix, multiplier in units:
        if lower.endswith(suffix):
            number = float(lower[: -len(suffix)])
            return int(number * multiplier)
    return int(lower)


def _format_size_bytes(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    if value % (1024 ** 3) == 0:
        return f"{value // (1024 ** 3)}GiB"
    if value % (1024 ** 2) == 0:
        return f"{value // (1024 ** 2)}MiB"
    return str(value)


class NativeQwenASRAdapter:
    """OfflineTranscriber-compatible adapter backed by the shared native ASR model."""

    def __init__(self, get_model, scheduler: ASRScheduler):
        self.get_model = get_model
        self.scheduler = scheduler

    def asr_waveform(self, wav: np.ndarray, sr: int = 16000, context: str = ""):
        if sr != 16000:
            raise ValueError("NativeQwenASRAdapter expects 16 kHz audio.")
        pcm = np.asarray(wav, dtype=np.float32).reshape(-1)
        def call():
            return self.get_model().transcribe(
                audio=[(pcm, sr)],
                context=[context or ""],
                return_time_stamps=False,
            )

        outputs = self.scheduler.run_sync(
            call,
            priority=ASRScheduler.PRIORITY_OFFLINE,
            label="offline-transcribe",
        )
        result = outputs[0]
        return getattr(result, "language", "") or "", getattr(result, "text", "") or ""


def _forced_aligner_config() -> dict[str, Any]:
    if _aligner_mode == "remote":
        return {
            "aligner_base_url": _aligner_base_url,
            "aligner_api_key": _aligner_api_key,
            "aligner_model": _aligner_model_path,
            "aligner_timeout_s": _aligner_timeout_s,
            "aligner_timestamp_segment_time_ms": _aligner_timestamp_segment_time_ms,
        }
    return {
        "aligner_base_url": "",
        "aligner_api_key": "",
        "aligner_model": "",
        "aligner_timeout_s": _aligner_timeout_s,
        "aligner_timestamp_segment_time_ms": _aligner_timestamp_segment_time_ms,
    }


def _build_offline_transcriber():
    global _offline_transcriber
    if _offline_transcriber is not None:
        return _offline_transcriber
    scheduler = _scheduler_required()
    adapter = NativeQwenASRAdapter(_load_model, scheduler)
    aligner_kwargs = _forced_aligner_config()
    _offline_transcriber = OfflineTranscriber(
        asr_client=adapter,
        num_threads=_offline_num_threads,
        vad_target_segment_s=_vad_target_segment_s,
        vad_max_segment_s=_vad_max_segment_s,
        **aligner_kwargs,
    )
    return _offline_transcriber


def _load_model():
    """Load Qwen3ASRModel via vLLM Python API."""
    global _qwen_asr_model
    if _qwen_asr_model is not None:
        return _qwen_asr_model

    from qwen_asr import Qwen3ASRModel

    model_path = _model_path or os.environ.get("QWEN3_ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B")
    print(f"[Model] Loading Qwen3-ASR from: {model_path}")
    llm_kwargs: dict[str, Any] = {
        "gpu_memory_utilization": _gpu_memory_utilization,
        "max_new_tokens": _max_new_tokens,
        "cpu_offload_gb": _cpu_offload_gb,
        "dtype": _dtype,
        "max_model_len": _max_model_len,
        "enforce_eager": _enforce_eager,
    }
    if _kv_cache_memory_bytes is not None:
        llm_kwargs["kv_cache_memory_bytes"] = _kv_cache_memory_bytes

    _qwen_asr_model = Qwen3ASRModel.LLM(
        model=model_path,
        **llm_kwargs,
    )
    print("[Model] Loaded successfully.")
    return _qwen_asr_model


def _capabilities() -> dict[str, Any]:
    return {
        "offline_http": bool(_enable_offline_api),
        "native_websocket": True,
        "forced_aligner": _aligner_mode,
        "max_concurrent_asr_jobs": _max_concurrent_asr_jobs,
        "requested_max_concurrent_asr_jobs": _requested_max_concurrent_asr_jobs,
        "asr_scheduler": "priority-single-worker",
        "realtime_priority": True,
        "asr_scheduler_queue_size": _asr_scheduler.qsize() if _asr_scheduler is not None else None,
        "asr_model_loaded_once": _qwen_asr_model is not None,
        "offline_num_threads": _offline_num_threads,
        "vad_target_segment_s": _vad_target_segment_s,
        "vad_max_segment_s": _vad_max_segment_s,
        "gpu_memory_utilization": _gpu_memory_utilization,
        "kv_cache_memory_bytes": _kv_cache_memory_bytes,
        "kv_cache_memory_human": _format_size_bytes(_kv_cache_memory_bytes),
        "cpu_offload_gb": _cpu_offload_gb,
        "max_model_len": _max_model_len,
        "dtype": _dtype,
        "enforce_eager": _enforce_eager,
    }


def _probe_model() -> dict:
    """Probe model readiness without a real inference (lightweight)."""
    if _qwen_asr_model is None:
        return {
            "status": "unavailable",
            "model": _model_path,
            "message": "Model not loaded yet.",
            "capabilities": _capabilities(),
        }

    try:
        # Try a dummy 0.1s silent audio to verify the pipeline works
        sr = 16000
        dummy = np.zeros(int(sr * 0.1), dtype=np.float32)
        def call() -> None:
            state = _qwen_asr_model.init_streaming_state(
                chunk_size_sec=0.1,
                unfixed_chunk_num=0,
                unfixed_token_num=0,
            )
            _qwen_asr_model.streaming_transcribe(dummy, state)
            _qwen_asr_model.finish_streaming_transcribe(state)

        _scheduler_required().run_sync(
            call,
            priority=ASRScheduler.PRIORITY_REALTIME,
            label="health-probe",
        )
        return {
            "status": "ok",
            "model": _model_path,
            "backend": "vllm",
            "message": "Model is ready.",
            "capabilities": _capabilities(),
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "model": _model_path,
            "message": f"Model loaded but probe failed: {exc}",
            "capabilities": _capabilities(),
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: attempt to load model
    try:
        _load_model()
        if _enable_offline_api:
            _build_offline_transcriber()
            print("[Offline] Native offline HTTP API is enabled.")
    except Exception as exc:
        print(f"[Warning] Model auto-load failed: {exc}")
        print("[Warning] /health will report unavailable until model is loaded.")
    yield
    # Shutdown
    if _asr_scheduler is not None:
        _asr_scheduler.shutdown()
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


@app.post("/api/v1/offline/transcribe")
async def offline_transcribe(
    audio_file: UploadFile = File(...),
    context: str = Form(default=""),
    use_forced_aligner: bool = Form(default=False),
):
    if not _enable_offline_api:
        return JSONResponse(
            content={"error": "Offline HTTP API is disabled."},
            status_code=404,
        )
    if _qwen_asr_model is None:
        return JSONResponse(
            content={"error": "Model not loaded. Check /health."},
            status_code=503,
        )
    if use_forced_aligner and _aligner_mode == "local-lazy":
        return JSONResponse(
            content={
                "error": "Local lazy forced aligner is not implemented in the unified native server yet.",
                "forced_aligner": {
                    "requested": True,
                    "available": False,
                    "message": "Use --aligner-mode remote or disabled.",
                },
            },
            status_code=400,
        )

    suffix = os.path.splitext(audio_file.filename or "upload.wav")[1] or ".wav"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            tmp.write(await audio_file.read())

        transcriber = _build_offline_transcriber()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: transcriber.transcribe_file(
                input_file=temp_path,
                context=context,
                use_forced_aligner=bool(use_forced_aligner),
            ),
        )
        result["file_name"] = audio_file.filename
        result["backend"] = "native-vllm"
        result["asr_model_loaded_once"] = True
        if use_forced_aligner and _aligner_mode == "disabled":
            result["forced_aligner"] = {
                "requested": True,
                "available": False,
                "message": "Forced aligner is disabled on this unified server.",
                "language": result.get("language", ""),
                "items": [],
                "text": result.get("text", ""),
            }
        return result
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Offline transcription failed: {exc}"},
            status_code=500,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


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
        def _locked_call():
            return func(*args)

        return await _scheduler_required().run_async(
            _locked_call,
            priority=ASRScheduler.PRIORITY_REALTIME,
            label=f"ws-{connection_id}",
            timeout=_decode_timeout_sec,
        )

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
                _scheduler_required().run_sync(
                    lambda: asr.finish_streaming_transcribe(state),
                    priority=ASRScheduler.PRIORITY_REALTIME,
                    label=f"ws-{connection_id}-cleanup",
                )
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
        default=float(os.environ.get("QWEN3_ASR_GPU_MEMORY_UTILIZATION", "0.5")),
        help="vLLM GPU memory utilization. Low-memory default is 0.5.",
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=str,
        default=os.environ.get("QWEN3_ASR_KV_CACHE_MEMORY_BYTES", ""),
        help="Exact vLLM KV cache limit, e.g. 8GiB or 8192MiB. Empty uses vLLM default.",
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=float(os.environ.get("QWEN3_ASR_CPU_OFFLOAD_GB", "0")),
        help="vLLM CPU offload in GiB.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=os.environ.get("QWEN3_ASR_DTYPE", "auto"),
        help="vLLM dtype, e.g. auto, bfloat16, float16.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("QWEN3_ASR_MAX_MODEL_LEN", "65536")),
        help="vLLM max model length.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=os.environ.get("QWEN3_ASR_ENFORCE_EAGER", "0") in {"1", "true", "TRUE", "yes", "YES"},
        help="Disable CUDA graph capture for lower-memory troubleshooting.",
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
    parser.add_argument(
        "--enable-offline-api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable POST /api/v1/offline/transcribe on the shared native ASR model.",
    )
    parser.add_argument(
        "--offline-num-threads",
        type=int,
        default=1,
        help="Offline segment worker count. Defaults to 1 to avoid competing with realtime WS.",
    )
    parser.add_argument(
        "--vad-target-segment-s",
        type=int,
        default=45,
        help="Offline VAD target segment length in seconds.",
    )
    parser.add_argument(
        "--vad-max-segment-s",
        type=int,
        default=60,
        help="Offline maximum segment length in seconds.",
    )
    parser.add_argument(
        "--aligner-mode",
        choices=["disabled", "remote", "local-lazy"],
        default=os.environ.get("QWEN3_ALIGNER_MODE", "disabled"),
        help="Forced aligner mode. local-lazy is reserved and returns an explicit error in this server.",
    )
    parser.add_argument(
        "--aligner-model-path",
        type=str,
        default=os.environ.get("QWEN3_ALIGNER_MODEL_PATH", "models/Qwen3-ForcedAligner-0.6B"),
        help="Forced aligner model path/name for remote or future local-lazy mode.",
    )
    parser.add_argument(
        "--aligner-base-url",
        type=str,
        default=os.environ.get("QWEN3_ALIGNER_BASE_URL", ""),
        help="Remote forced aligner base URL when --aligner-mode=remote.",
    )
    parser.add_argument(
        "--aligner-api-key",
        type=str,
        default=os.environ.get("QWEN3_ALIGNER_API_KEY", "EMPTY"),
        help="Remote forced aligner API key when --aligner-mode=remote.",
    )
    parser.add_argument(
        "--aligner-timeout-s",
        type=int,
        default=int(os.environ.get("QWEN3_ALIGNER_TIMEOUT_S", "120")),
        help="Remote forced aligner timeout in seconds.",
    )
    parser.add_argument(
        "--aligner-timestamp-segment-time-ms",
        type=int,
        default=int(os.environ.get("QWEN3_ALIGNER_TIMESTAMP_SEGMENT_TIME_MS", "80")),
        help="Forced aligner timestamp smoothing minimum segment duration in ms.",
    )
    parser.add_argument(
        "--max-concurrent-asr-jobs",
        type=int,
        default=1,
        help="Compatibility option. Native model execution is serialized through a realtime-priority scheduler.",
    )
    return parser.parse_args()


def main():
    global _model_path, _gpu_memory_utilization, _max_new_tokens
    global _kv_cache_memory_bytes, _cpu_offload_gb, _dtype, _max_model_len, _enforce_eager
    global _default_chunk_size_sec, _default_unfixed_chunk_num, _default_unfixed_token_num
    global _audio_queue_size, _send_queue_size, _decode_timeout_sec
    global _enable_offline_api, _offline_num_threads, _vad_target_segment_s, _vad_max_segment_s
    global _aligner_mode, _aligner_model_path, _aligner_base_url, _aligner_api_key
    global _aligner_timeout_s, _aligner_timestamp_segment_time_ms
    global _max_concurrent_asr_jobs, _requested_max_concurrent_asr_jobs, _asr_scheduler, _offline_transcriber

    args = parse_args()
    _model_path = args.model_path or os.environ.get("QWEN3_ASR_MODEL_PATH")
    _gpu_memory_utilization = args.gpu_memory_utilization
    _kv_cache_memory_bytes = _parse_size_bytes(args.kv_cache_memory_bytes)
    _cpu_offload_gb = float(args.cpu_offload_gb)
    _dtype = args.dtype
    _max_model_len = int(args.max_model_len)
    _enforce_eager = bool(args.enforce_eager)
    _max_new_tokens = args.max_new_tokens
    _default_chunk_size_sec = args.chunk_size_sec
    _default_unfixed_chunk_num = args.unfixed_chunk_num
    _default_unfixed_token_num = args.unfixed_token_num
    _audio_queue_size = max(1, args.audio_queue_size)
    _send_queue_size = max(1, args.send_queue_size)
    _decode_timeout_sec = max(0.0, args.decode_timeout_sec)
    _enable_offline_api = bool(args.enable_offline_api)
    _offline_num_threads = max(1, args.offline_num_threads)
    _vad_target_segment_s = max(1, args.vad_target_segment_s)
    _vad_max_segment_s = max(1, args.vad_max_segment_s)
    _aligner_mode = args.aligner_mode
    _aligner_model_path = args.aligner_model_path
    _aligner_base_url = args.aligner_base_url
    _aligner_api_key = args.aligner_api_key
    _aligner_timeout_s = int(args.aligner_timeout_s)
    _aligner_timestamp_segment_time_ms = int(args.aligner_timestamp_segment_time_ms)
    _requested_max_concurrent_asr_jobs = max(1, args.max_concurrent_asr_jobs)
    _max_concurrent_asr_jobs = 1
    if _requested_max_concurrent_asr_jobs != 1:
        print("[ASR Scheduler] --max-concurrent-asr-jobs is kept for compatibility; native model execution is serialized with realtime priority.")
    _asr_scheduler = ASRScheduler()
    _offline_transcriber = None

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
