from __future__ import annotations

import numpy as np
from librosa import resample

TARGET_SAMPLE_RATE = 16000


def to_mono_float32(audio: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
    sample_rate, data = audio
    wav = np.asarray(data)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    elif wav.ndim > 2:
        wav = wav.reshape(-1)

    if np.issubdtype(wav.dtype, np.integer):
        max_abs = max(abs(np.iinfo(wav.dtype).min), np.iinfo(wav.dtype).max)
        wav = wav.astype(np.float32) / float(max_abs)
    else:
        wav = wav.astype(np.float32)
        max_value = float(np.max(np.abs(wav))) if wav.size else 0.0
        if max_value > 1.5:
            wav = wav / max_value
    return sample_rate, wav.astype(np.float32, copy=False)


def resample_to_16k(wav: np.ndarray, source_sr: int) -> np.ndarray:
    if source_sr == TARGET_SAMPLE_RATE:
        return wav.astype(np.float32, copy=False)
    if wav.size == 0:
        return wav.astype(np.float32, copy=False)
    return resample(y=wav.astype(np.float32), orig_sr=source_sr, target_sr=TARGET_SAMPLE_RATE).astype(np.float32)


def ensure_float32_pcm_16k(audio: tuple[int, np.ndarray]) -> np.ndarray:
    sample_rate, wav = to_mono_float32(audio)
    return resample_to_16k(wav, sample_rate)
