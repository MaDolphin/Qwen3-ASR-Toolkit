import asyncio
import argparse
import json
import os
import tempfile
import uuid

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect

from qwen3_asr_toolkit.audio_tools import WAV_SAMPLE_RATE
from qwen3_asr_toolkit.env_utils import load_project_dotenv
from qwen3_asr_toolkit.offline_transcriber import OfflineTranscriber
from qwen3_asr_toolkit.qwen3asr import QwenASR
from qwen3_asr_toolkit.streaming_transcriber import StreamingSession, StreamingTranscriber


def create_app() -> FastAPI:
    load_project_dotenv()
    app = FastAPI(title="Qwen3-ASR-Toolkit Server", version="2.0.0")

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    model = os.environ.get("QWEN3_ASR_MODEL", "qwen3-asr-flash")
    num_threads = int(os.environ.get("QWEN3_ASR_NUM_THREADS", "4"))
    vad_target_segment_s = int(os.environ.get("QWEN3_ASR_VAD_TARGET_SEGMENT_S", "45"))
    vad_max_segment_s = int(os.environ.get("QWEN3_ASR_VAD_MAX_SEGMENT_S", "60"))
    aligner_base_url = os.environ.get("QWEN3_ALIGNER_BASE_URL", "")
    aligner_api_key = os.environ.get("QWEN3_ALIGNER_API_KEY", api_key)
    aligner_model = os.environ.get("QWEN3_ALIGNER_MODEL", "Qwen3-ForcedAligner-0.6B")
    aligner_timeout_s = int(os.environ.get("QWEN3_ALIGNER_TIMEOUT_S", "120"))
    aligner_timestamp_segment_time_ms = int(os.environ.get("QWEN3_ALIGNER_TIMESTAMP_SEGMENT_TIME_MS", "80"))
    decode_interval_ms = int(float(os.environ.get("QWEN3_ASR_STREAM_DECODE_STRIDE_S", "1.0")) * 1000)
    decode_interval_ms = int(os.environ.get("QWEN3_ASR_STREAM_DECODE_INTERVAL_MS", str(decode_interval_ms)))
    min_chunk_ms = int(os.environ.get("QWEN3_ASR_STREAM_MIN_CHUNK_MS", "200"))
    finalize_silence_ms = int(os.environ.get("QWEN3_ASR_STREAM_FINALIZE_SILENCE_MS", "600"))
    max_segment_sec = float(os.environ.get("QWEN3_ASR_STREAM_MAX_SEGMENT_S", "20"))

    asr_client = QwenASR(model=model, base_url=base_url, api_key=api_key)
    offline_transcriber = OfflineTranscriber(
        asr_client=asr_client,
        num_threads=num_threads,
        vad_target_segment_s=vad_target_segment_s,
        vad_max_segment_s=vad_max_segment_s,
        aligner_base_url=aligner_base_url,
        aligner_api_key=aligner_api_key,
        aligner_model=aligner_model,
        aligner_timeout_s=aligner_timeout_s,
        aligner_timestamp_segment_time_ms=aligner_timestamp_segment_time_ms,
    )
    streaming_transcriber = StreamingTranscriber(
        offline_transcriber=offline_transcriber,
        decode_interval_ms=decode_interval_ms,
        min_chunk_ms=min_chunk_ms,
        finalize_silence_ms=finalize_silence_ms,
        max_segment_sec=max_segment_sec,
    )

    app.state.offline_transcriber = offline_transcriber
    app.state.streaming_transcriber = streaming_transcriber
    app.state.server_config = {
        "model": model,
        "sample_rate": WAV_SAMPLE_RATE,
        "vad_max_segment_s": vad_max_segment_s,
        "stream_decode_interval_ms": decode_interval_ms,
        "stream_min_chunk_ms": min_chunk_ms,
        "stream_finalize_silence_ms": finalize_silence_ms,
        "stream_max_segment_sec": max_segment_sec,
    }

    @app.get("/health")
    async def health():
        return {"status": "ok", **app.state.server_config}

    @app.post("/api/v1/offline/transcribe")
    async def offline_transcribe(
        audio_file: UploadFile = File(...),
        context: str = Form(default=""),
        use_forced_aligner: bool = Form(default=False),
    ):
        suffix = os.path.splitext(audio_file.filename or "upload.wav")[1] or ".wav"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = tmp.name
                tmp.write(await audio_file.read())

            result = app.state.offline_transcriber.transcribe_file(
                input_file=temp_path,
                context=context,
                use_forced_aligner=use_forced_aligner,
            )
            result["file_name"] = audio_file.filename
            return result
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    @app.websocket("/ws/v1/realtime/transcribe")
    async def realtime_transcribe(websocket: WebSocket):
        await websocket.accept()
        session = StreamingSession(session_id=uuid.uuid4().hex)
        queue: asyncio.Queue = asyncio.Queue()

        await websocket.send_json(
            {
                "event": "ready",
                "protocol_version": "2.0",
                "session_id": session.session_id,
                "sample_rate": WAV_SAMPLE_RATE,
                "audio_format": "float32le_pcm_mono_16k",
                "realtime_alignment_supported": False,
                "session_defaults": {
                    "decode_interval_ms": app.state.server_config["stream_decode_interval_ms"],
                    "min_chunk_ms": app.state.server_config["stream_min_chunk_ms"],
                    "finalize_silence_ms": app.state.server_config["stream_finalize_silence_ms"],
                    "max_segment_sec": app.state.server_config["stream_max_segment_sec"],
                },
            }
        )

        async def _receiver_loop():
            try:
                while True:
                    message = await websocket.receive()
                    await queue.put(message)
                    if message.get("type") == "websocket.disconnect":
                        break
            except WebSocketDisconnect:
                await queue.put({"type": "websocket.disconnect"})
            except RuntimeError:
                await queue.put({"type": "websocket.disconnect"})

        async def _worker_loop():
            started = False
            loop = asyncio.get_running_loop()
            while True:
                message = await queue.get()
                if message.get("type") == "websocket.disconnect":
                    if not session.closed:
                        try:
                            await loop.run_in_executor(None, app.state.streaming_transcriber.finish, session)
                        except Exception:
                            pass
                    break

                if "text" in message and message["text"] is not None:
                    try:
                        payload = json.loads(message["text"])
                    except Exception:
                        await websocket.send_json({"event": "error", "message": "Invalid JSON message."})
                        continue

                    event = payload.get("event", "")
                    if event == "start":
                        if payload.get("use_forced_aligner") or payload.get("aligner") or payload.get("forced_aligner"):
                            await websocket.send_json(
                                {
                                    "event": "error",
                                    "message": "Realtime transcription does not support forced aligner.",
                                }
                            )
                            continue

                        session.session_id = payload.get("session_id", session.session_id) or session.session_id
                        started = True
                        started_payload = app.state.streaming_transcriber.configure_session(
                            session,
                            context=payload.get("context", "") or "",
                            decode_interval_ms=payload.get("decode_interval_ms"),
                            min_chunk_ms=payload.get("min_chunk_ms"),
                            finalize_silence_ms=payload.get("finalize_silence_ms"),
                            max_segment_sec=payload.get("max_segment_sec"),
                        )
                        await websocket.send_json(started_payload)
                    elif event == "finish":
                        final = await loop.run_in_executor(None, app.state.streaming_transcriber.finish, session)
                        await websocket.send_json(final)
                        await websocket.close()
                        break
                    elif event == "ping":
                        await websocket.send_json({"event": "pong"})
                    else:
                        await websocket.send_json({"event": "error", "message": f"Unsupported event: {event}"})
                elif "bytes" in message and message["bytes"] is not None:
                    if not started:
                        await websocket.send_json({"event": "error", "message": "Send a start event before audio chunks."})
                        continue

                    raw = message["bytes"]
                    if len(raw) % 4 != 0:
                        await websocket.send_json(
                            {"event": "error", "message": "Audio chunk must be float32 bytes (multiple of 4)."}
                        )
                        continue

                    chunk = np.frombuffer(raw, dtype=np.float32).reshape(-1).copy()
                    partial = await loop.run_in_executor(None, app.state.streaming_transcriber.push_audio, session, chunk)
                    await websocket.send_json(partial)
                else:
                    await websocket.send_json({"event": "error", "message": "Unsupported websocket frame."})

        receiver_task = asyncio.create_task(_receiver_loop())
        worker_task = asyncio.create_task(_worker_loop())
        try:
            await asyncio.gather(receiver_task, worker_task)
        finally:
            for task in (receiver_task, worker_task):
                if not task.done():
                    task.cancel()

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR toolkit server (REST + WebSocket).")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=18000, help="Bind port.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload.")
    return parser.parse_args()


def main():
    args = parse_args()
    uvicorn.run(
        "qwen3_asr_toolkit.server:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
