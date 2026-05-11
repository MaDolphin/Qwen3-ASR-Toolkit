from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import websockets


class RealtimeClientError(RuntimeError):
    pass


@dataclass
class RealtimeMetrics:
    chunks_sent: int = 0
    ack_count: int = 0
    partial_count: int = 0
    final_count: int = 0
    error_count: int = 0
    started: bool = False
    last_partial: str = ""
    final_text: str = ""
    language: str = ""
    last_event: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "chunks_sent": self.chunks_sent,
            "ack_count": self.ack_count,
            "partial_count": self.partial_count,
            "final_count": self.final_count,
            "error_count": self.error_count,
            "language": self.language,
            "last_partial_len": len(self.last_partial),
            "final_text_len": len(self.final_text),
            "last_event": self.last_event,
        }


class RealtimeWSClient:
    def __init__(
        self,
        ws_url: str,
        context: str = "",
        chunk_size_sec: float = 1.0,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        receive_timeout_sec: float = 300.0,
    ):
        self.ws_url = ws_url
        self.context = context
        self.chunk_size_sec = chunk_size_sec
        self.unfixed_chunk_num = unfixed_chunk_num
        self.unfixed_token_num = unfixed_token_num
        self.receive_timeout_sec = receive_timeout_sec
        self.metrics = RealtimeMetrics()
        self._websocket = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def build_start_payload(self) -> dict[str, Any]:
        return {
            "event": "start",
            "stream": True,
            "context": self.context,
            "chunk_size_sec": self.chunk_size_sec,
            "unfixed_chunk_num": self.unfixed_chunk_num,
            "unfixed_token_num": self.unfixed_token_num,
        }

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None and self._loop.is_running():
            return self._loop
        self._loop = asyncio.new_event_loop()

        def run_loop() -> None:
            assert self._loop is not None
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        return self._loop

    def _run(self, coro):
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=self.receive_timeout_sec + 30.0)

    async def _ensure_started(self) -> dict[str, Any]:
        if self._websocket is not None and self.metrics.started:
            return self.metrics.last_event
        self._websocket = await websockets.connect(self.ws_url, max_size=None, ping_interval=None)
        await self._websocket.send(json.dumps(self.build_start_payload(), ensure_ascii=False))
        message = await asyncio.wait_for(self._websocket.recv(), timeout=self.receive_timeout_sec)
        event = json.loads(message)
        self.metrics.last_event = event
        if event.get("event") == "error":
            raise RealtimeClientError(event.get("message", "websocket start failed"))
        if event.get("event") != "started":
            raise RealtimeClientError(f"未收到 started 事件：{event}")
        self.metrics.started = True
        return event

    async def _drain_until_ack_or_final(self) -> None:
        if self._websocket is None:
            raise RealtimeClientError("WebSocket 尚未启动。")
        while True:
            message = await asyncio.wait_for(self._websocket.recv(), timeout=self.receive_timeout_sec)
            event = json.loads(message)
            event_name = event.get("event")
            self.metrics.last_event = event
            if event_name == "ack":
                self.metrics.ack_count += 1
                return
            if event_name == "partial":
                self.metrics.partial_count += 1
                self.metrics.last_partial = event.get("text") or self.metrics.last_partial
                self.metrics.language = event.get("language") or self.metrics.language
                continue
            if event_name == "final":
                self.metrics.final_count += 1
                self.metrics.final_text = event.get("text") or ""
                self.metrics.language = event.get("language") or self.metrics.language
                return
            if event_name == "error":
                self.metrics.error_count += 1
                raise RealtimeClientError(event.get("message", "server error"))

    async def _send_audio(self, pcm_float32_16k: np.ndarray) -> dict[str, Any]:
        await self._ensure_started()
        if self._websocket is None:
            raise RealtimeClientError("WebSocket 尚未启动。")
        pcm = np.asarray(pcm_float32_16k, dtype=np.float32)
        if pcm.size == 0:
            return self.metrics.as_dict()
        await self._websocket.send(pcm.tobytes())
        self.metrics.chunks_sent += 1
        await self._drain_until_ack_or_final()
        return self.metrics.as_dict()

    async def _finish(self) -> dict[str, Any]:
        await self._ensure_started()
        if self._websocket is None:
            raise RealtimeClientError("WebSocket 尚未启动。")
        await self._websocket.send(json.dumps({"event": "finish"}, ensure_ascii=False))
        while True:
            message = await asyncio.wait_for(self._websocket.recv(), timeout=self.receive_timeout_sec)
            event = json.loads(message)
            event_name = event.get("event")
            self.metrics.last_event = event
            if event_name == "partial":
                self.metrics.partial_count += 1
                self.metrics.last_partial = event.get("text") or self.metrics.last_partial
                self.metrics.language = event.get("language") or self.metrics.language
                continue
            if event_name == "final":
                self.metrics.final_count += 1
                self.metrics.final_text = event.get("text") or ""
                self.metrics.language = event.get("language") or self.metrics.language
                await self._close_ws()
                return self.metrics.as_dict()
            if event_name == "error":
                self.metrics.error_count += 1
                raise RealtimeClientError(event.get("message", "server error"))

    async def _close_ws(self) -> None:
        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None

    def start(self) -> dict[str, Any]:
        return self._run(self._ensure_started())

    def send_audio(self, pcm_float32_16k: np.ndarray) -> dict[str, Any]:
        return self._run(self._send_audio(pcm_float32_16k))

    def finish(self) -> dict[str, Any]:
        result = self._run(self._finish())
        self.close()
        return result

    def close(self) -> None:
        if self._loop is not None and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._close_ws(), self._loop)
                future.result(timeout=5.0)
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._thread is not None:
                    self._thread.join(timeout=5.0)
        self._loop = None
        self._thread = None
