from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from qwen3_asr_toolkit.audio_tools import WAV_SAMPLE_RATE


def _ms_to_samples(value_ms: int) -> int:
    return max(1, int(round((float(value_ms) / 1000.0) * WAV_SAMPLE_RATE)))


def _concat_text(left: str, right: str) -> str:
    left = left or ""
    right = right or ""
    if not left:
        return right
    if not right:
        return left
    if left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum():
        return f"{left} {right}"
    return left + right


@dataclass
class StreamingSession:
    context: str = ""
    session_id: Optional[str] = None
    segment_audio: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    committed_text: str = ""
    live_text: str = ""
    language: str = ""
    silence_samples: int = 0
    received_samples: int = 0
    segment_index: int = 0
    decode_count: int = 0
    pending_decode_samples: int = 0
    decode_interval_samples: int = _ms_to_samples(600)
    min_chunk_samples: int = _ms_to_samples(200)
    finalize_silence_samples: int = _ms_to_samples(600)
    max_segment_samples: int = int(20 * WAV_SAMPLE_RATE)
    speech_threshold: float = 0.008
    configured: bool = False
    closed: bool = False

    @property
    def text(self) -> str:
        return _concat_text(self.committed_text, self.live_text)

    @text.setter
    def text(self, value: str) -> None:
        self.committed_text = value or ""
        self.live_text = ""


class StreamingTranscriber:
    def __init__(
        self,
        offline_transcriber,
        decode_interval_ms: int = 600,
        min_chunk_ms: int = 200,
        finalize_silence_ms: int = 600,
        max_segment_sec: float = 20.0,
        speech_threshold: float = 0.008,
    ):
        self.offline_transcriber = offline_transcriber
        self.default_decode_interval_ms = int(decode_interval_ms)
        self.default_min_chunk_ms = int(min_chunk_ms)
        self.default_finalize_silence_ms = int(finalize_silence_ms)
        self.default_max_segment_sec = float(max_segment_sec)
        self.default_speech_threshold = float(speech_threshold)

    def configure_session(
        self,
        session: StreamingSession,
        *,
        context: str = "",
        decode_interval_ms: Optional[int] = None,
        min_chunk_ms: Optional[int] = None,
        finalize_silence_ms: Optional[int] = None,
        max_segment_sec: Optional[float] = None,
    ) -> Dict:
        if session.closed:
            raise ValueError("Streaming session is already closed.")

        session.context = context or ""
        session.decode_interval_samples = _ms_to_samples(decode_interval_ms or self.default_decode_interval_ms)
        session.min_chunk_samples = _ms_to_samples(min_chunk_ms or self.default_min_chunk_ms)
        session.finalize_silence_samples = _ms_to_samples(finalize_silence_ms or self.default_finalize_silence_ms)
        session.max_segment_samples = max(
            1,
            int(round(float(max_segment_sec or self.default_max_segment_sec) * WAV_SAMPLE_RATE)),
        )
        session.speech_threshold = self.default_speech_threshold
        session.configured = True
        return {
            "event": "started",
            "session_id": session.session_id,
            "session": {
                "context": session.context,
                "decode_interval_ms": int(round(session.decode_interval_samples * 1000.0 / WAV_SAMPLE_RATE)),
                "min_chunk_ms": int(round(session.min_chunk_samples * 1000.0 / WAV_SAMPLE_RATE)),
                "finalize_silence_ms": int(round(session.finalize_silence_samples * 1000.0 / WAV_SAMPLE_RATE)),
                "max_segment_sec": round(session.max_segment_samples / WAV_SAMPLE_RATE, 3),
            },
        }

    def push_audio(self, session: StreamingSession, pcm16k: np.ndarray) -> Dict:
        if session.closed:
            raise ValueError("Streaming session is already closed.")
        self._ensure_session_defaults(session)

        x = self._normalize_audio(pcm16k)
        if x.size == 0:
            return self._build_partial(session, updated=False)

        session.segment_audio = np.concatenate([session.segment_audio, x], axis=0)
        session.received_samples += len(x)
        session.pending_decode_samples += len(x)
        self._update_silence(session, x)

        if self._should_finalize_segment(session):
            segment_audio = self._trim_trailing_silence(session.segment_audio, session.silence_samples)
            decoded = self._decode_audio(segment_audio, context=session.context)
            return self._commit_segment(session, decoded)

        should_decode = (
            len(session.segment_audio) >= session.min_chunk_samples
            and session.pending_decode_samples >= session.decode_interval_samples
        )
        if should_decode:
            decoded = self._decode_audio(session.segment_audio, context=session.context)
            session.language = decoded["language"]
            session.live_text = decoded["text"]
            session.pending_decode_samples = 0
            session.decode_count += 1

        return self._build_partial(session, updated=should_decode)

    def finish(self, session: StreamingSession) -> Dict:
        if session.closed:
            return self._build_final(session)
        self._ensure_session_defaults(session)

        remaining = self._trim_trailing_silence(session.segment_audio, session.silence_samples)
        if remaining.size > 0:
            decoded = self._decode_audio(remaining, context=session.context)
            session.language = decoded["language"]
            session.committed_text = _concat_text(session.committed_text, decoded["text"])
            session.live_text = ""
            session.segment_audio = np.zeros((0,), dtype=np.float32)
            session.pending_decode_samples = 0
            session.silence_samples = 0
            session.decode_count += 1

        session.closed = True
        return self._build_final(session)

    def _normalize_audio(self, pcm16k: np.ndarray) -> np.ndarray:
        x = np.asarray(pcm16k)
        if x.ndim != 1:
            x = x.reshape(-1)
        if x.dtype == np.int16:
            return x.astype(np.float32) / 32768.0
        return x.astype(np.float32, copy=False)

    def _ensure_session_defaults(self, session: StreamingSession) -> None:
        if session.configured:
            return
        session.decode_interval_samples = _ms_to_samples(self.default_decode_interval_ms)
        session.min_chunk_samples = _ms_to_samples(self.default_min_chunk_ms)
        session.finalize_silence_samples = _ms_to_samples(self.default_finalize_silence_ms)
        session.max_segment_samples = max(1, int(round(self.default_max_segment_sec * WAV_SAMPLE_RATE)))
        session.speech_threshold = self.default_speech_threshold
        session.configured = True

    def _update_silence(self, session: StreamingSession, chunk: np.ndarray) -> None:
        if self._is_silence(chunk, threshold=session.speech_threshold):
            session.silence_samples += len(chunk)
        else:
            session.silence_samples = 0

    def _is_silence(self, wav: np.ndarray, threshold: float) -> bool:
        if wav.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(wav.astype(np.float32, copy=False)))))
        return rms < float(threshold)

    def _trim_trailing_silence(self, wav: np.ndarray, silence_samples: int) -> np.ndarray:
        if wav.size == 0:
            return wav
        if silence_samples <= 0 or silence_samples >= wav.size:
            return wav
        return wav[:-silence_samples]

    def _should_finalize_segment(self, session: StreamingSession) -> bool:
        active_audio = self._trim_trailing_silence(session.segment_audio, session.silence_samples)
        if active_audio.size == 0:
            return False
        if session.silence_samples >= session.finalize_silence_samples:
            return True
        return len(session.segment_audio) >= session.max_segment_samples

    def _decode_audio(self, wav: np.ndarray, context: str) -> Dict:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if wav.size == 0:
            return {"language": "", "text": ""}
        result = self.offline_transcriber.transcribe_waveform(
            wav,
            context=context,
            use_forced_aligner=False,
        )
        return {"language": result.get("language", ""), "text": result.get("text", "")}

    def _commit_segment(self, session: StreamingSession, decoded: Dict) -> Dict:
        segment_text = decoded["text"]
        session.language = decoded["language"] or session.language
        session.committed_text = _concat_text(session.committed_text, segment_text)
        session.live_text = ""
        session.segment_audio = np.zeros((0,), dtype=np.float32)
        session.pending_decode_samples = 0
        session.silence_samples = 0
        event = {
            "event": "segment_final",
            "language": session.language,
            "text": session.committed_text,
            "committed_text": session.committed_text,
            "live_text": "",
            "segment_text": segment_text,
            "segment_index": session.segment_index,
            "audio_duration_sec": round(session.received_samples / WAV_SAMPLE_RATE, 3),
        }
        session.segment_index += 1
        session.decode_count += 1
        return event

    def _build_partial(self, session: StreamingSession, updated: bool) -> Dict:
        return {
            "event": "partial",
            "updated": bool(updated),
            "language": session.language,
            "text": session.text,
            "committed_text": session.committed_text,
            "live_text": session.live_text,
            "segment_index": session.segment_index,
            "audio_duration_sec": round(session.received_samples / WAV_SAMPLE_RATE, 3),
        }

    def _build_final(self, session: StreamingSession) -> Dict:
        return {
            "event": "final",
            "language": session.language,
            "text": session.committed_text,
            "committed_text": session.committed_text,
            "live_text": "",
            "segment_index": session.segment_index,
            "audio_duration_sec": round(session.received_samples / WAV_SAMPLE_RATE, 3),
        }
