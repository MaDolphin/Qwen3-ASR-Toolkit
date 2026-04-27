"""
Shared utilities for Qwen3-ASR streaming servers.

Contains common helpers for:
  - Audio buffer accumulation and float32 validation
  - WebSocket event parsing and dispatch
  - ASR output post-processing (duplicate prefix cleanup)
"""

import json
import re
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
AUDIO_FORMAT = "float32le_pcm_mono_16k"


def validate_audio_chunk(raw_bytes: bytes) -> Tuple[bool, Optional[str], Optional[np.ndarray]]:
    """
    Validate a raw audio chunk and convert to float32 ndarray.

    Args:
        raw_bytes: Raw bytes received over WebSocket.

    Returns:
        Tuple of (ok, error_message, ndarray). If ok is False, ndarray is None.
    """
    if len(raw_bytes) % 4 != 0:
        return False, "Audio chunk must be float32 bytes (multiple of 4).", None

    chunk = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1).copy()
    if chunk.size == 0:
        return False, "Empty audio chunk.", None

    return True, None, chunk


def accumulate_buffer(buffer: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """Concatenate a new chunk onto the existing buffer."""
    return np.concatenate([buffer, chunk], axis=0)


def consume_full_chunks(
    buffer: np.ndarray, chunk_size_samples: int
) -> Tuple[np.ndarray, list]:
    """
    Consume as many full chunks as possible from the buffer.

    Args:
        buffer: Current audio buffer.
        chunk_size_samples: Number of samples per chunk.

    Returns:
        Tuple of (remaining_buffer, list_of_consumed_chunks).
    """
    consumed = []
    remaining = buffer
    while remaining.shape[0] >= chunk_size_samples:
        feed = remaining[:chunk_size_samples]
        remaining = remaining[chunk_size_samples:]
        consumed.append(feed)
    return remaining, consumed


# ---------------------------------------------------------------------------
# ASR output post-processing
# ---------------------------------------------------------------------------

_LANG_ASR_PREFIX_RE = re.compile(r"language\s+\S+\s*<asr_text>")


def clean_duplicate_asr_prefixes(text: str) -> str:
    """
    Remove repeated ``language {Lang}<asr_text>`` prefixes, keeping only the first.

    vLLM's SSE streaming endpoint for long audio can intermittently re-emit the
    language marker at chunk boundaries. This function deduplicates them client-side.

    Examples:
        >>> clean_duplicate_asr_prefixes(
        ...     "language German<asr_text>Hello language German<asr_text>World"
        ... )
        'language German<asr_text>Hello World'
    """
    matches = list(_LANG_ASR_PREFIX_RE.finditer(text))
    if len(matches) <= 1:
        return text

    # Keep everything up to and including the first match, then strip all
    # subsequent matches from the remainder.
    first_end = matches[0].end()
    prefix_part = text[:first_end]
    remainder = text[first_end:]
    cleaned_remainder = _LANG_ASR_PREFIX_RE.sub("", remainder)
    return prefix_part + cleaned_remainder


def build_ack_event(received_samples: int, total_samples: int) -> dict:
    """Build an ``ack`` event dict for WebSocket clients."""
    return {
        "event": "ack",
        "received_samples": received_samples,
        "total_samples": total_samples,
        "duration_sec": round(total_samples / SAMPLE_RATE, 3),
    }


def build_started_event(
    *,
    stream: bool = True,
    chunk_size_sec: Optional[float] = None,
    unfixed_chunk_num: Optional[int] = None,
    unfixed_token_num: Optional[int] = None,
) -> dict:
    """Build a ``started`` event dict with common fields."""
    evt = {
        "event": "started",
        "stream": stream,
        "sample_rate": SAMPLE_RATE,
        "audio_format": AUDIO_FORMAT,
    }
    if chunk_size_sec is not None:
        evt["chunk_size_sec"] = chunk_size_sec
    if unfixed_chunk_num is not None:
        evt["unfixed_chunk_num"] = unfixed_chunk_num
    if unfixed_token_num is not None:
        evt["unfixed_token_num"] = unfixed_token_num
    return evt
