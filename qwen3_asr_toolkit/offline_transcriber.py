import concurrent.futures
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from silero_vad import load_silero_vad

from qwen3_asr_toolkit.audio_tools import WAV_SAMPLE_RATE, load_audio, process_vad
from qwen3_asr_toolkit.forced_aligner_client import RemoteForcedAlignerClient


@dataclass
class AudioSegmentChunk:
    index: int
    start_sample: int
    end_sample: int
    wav: np.ndarray


def merge_languages(languages: List[str]) -> str:
    merged = []
    prev = None
    for language in languages:
        current = (language or "").strip()
        if not current:
            continue
        if prev == current:
            continue
        merged.append(current)
        prev = current
    return ",".join(merged)


class OfflineTranscriber:
    def __init__(
        self,
        asr_client,
        num_threads: int = 4,
        vad_target_segment_s: int = 45,
        vad_max_segment_s: int = 60,
        aligner_base_url: Optional[str] = None,
        aligner_api_key: Optional[str] = None,
        aligner_model: Optional[str] = None,
        aligner_timeout_s: int = 120,
        aligner_timestamp_segment_time_ms: int = 80,
    ):
        self.asr_client = asr_client
        self.num_threads = max(1, int(num_threads))
        self.vad_target_segment_s = max(1, int(vad_target_segment_s))
        self.vad_max_segment_s = max(1, int(vad_max_segment_s))
        self.aligner_base_url = (aligner_base_url or "").strip() or None
        self.aligner_api_key = (aligner_api_key or "").strip() or None
        self.aligner_model = (aligner_model or "").strip() or None
        self.aligner_timeout_s = int(aligner_timeout_s)
        self.aligner_timestamp_segment_time_ms = int(aligner_timestamp_segment_time_ms)
        self._vad_model = None
        self._aligner_client = None

    def transcribe_file(
        self,
        input_file: str,
        context: str = "",
        use_forced_aligner: bool = False,
    ) -> Dict:
        wav = load_audio(input_file)
        return self.transcribe_waveform(
            wav=wav,
            context=context,
            use_forced_aligner=use_forced_aligner,
        )

    def transcribe_waveform(
        self,
        wav: np.ndarray,
        context: str = "",
        use_forced_aligner: bool = False,
    ) -> Dict:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        chunks = self._build_chunks(wav)

        def _run_one(chunk: AudioSegmentChunk):
            language, text = self.asr_client.asr_waveform(chunk.wav, sr=WAV_SAMPLE_RATE, context=context)
            return chunk.index, language, text

        results: List[Tuple[int, str, str]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(_run_one, chunk) for chunk in chunks]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda item: item[0])

        segments = []
        languages = []
        texts = []
        for (idx, language, text), chunk in zip(results, chunks):
            languages.append(language)
            texts.append(text)
            segments.append(
                {
                    "index": idx,
                    "start_sec": round(chunk.start_sample / WAV_SAMPLE_RATE, 3),
                    "end_sec": round(chunk.end_sample / WAV_SAMPLE_RATE, 3),
                    "duration_sec": round((chunk.end_sample - chunk.start_sample) / WAV_SAMPLE_RATE, 3),
                    "language": language,
                    "text": text,
                }
            )

        forced_aligner = self._build_forced_aligner_metadata(
            requested=bool(use_forced_aligner),
            language=merge_languages(languages),
            text="".join(texts),
            segments=segments,
            chunks=chunks,
        )

        return {
            "language": merge_languages(languages),
            "text": "".join(texts),
            "segment_count": len(segments),
            "audio_duration_sec": round(len(wav) / WAV_SAMPLE_RATE, 3),
            "segments": segments,
            "forced_aligner": forced_aligner,
        }

    def _build_chunks(self, wav: np.ndarray) -> List[AudioSegmentChunk]:
        total_duration = len(wav) / WAV_SAMPLE_RATE
        if total_duration <= self.vad_max_segment_s:
            return [AudioSegmentChunk(index=0, start_sample=0, end_sample=len(wav), wav=wav)]

        try:
            if self._vad_model is None:
                self._vad_model = load_silero_vad(onnx=True)

            raw_chunks = process_vad(
                wav,
                self._vad_model,
                segment_threshold_s=min(self.vad_target_segment_s, self.vad_max_segment_s),
                max_segment_threshold_s=self.vad_max_segment_s,
            )
            normalized = self._normalize_raw_chunks(raw_chunks, wav)
            return self._enforce_max_duration(normalized)
        except Exception:
            return self._split_by_fixed_max(wav)

    def _normalize_raw_chunks(self, raw_chunks, wav: np.ndarray) -> List[AudioSegmentChunk]:
        total_samples = len(wav)
        chunks = []
        for idx, item in enumerate(raw_chunks):
            start_sample = int(max(0, item[0]))
            end_sample = int(min(total_samples, item[1]))
            if end_sample <= start_sample:
                continue
            chunk_wav = np.asarray(item[2], dtype=np.float32).reshape(-1)
            if len(chunk_wav) != (end_sample - start_sample):
                chunk_wav = np.asarray(
                    np.zeros((end_sample - start_sample,), dtype=np.float32)
                    if (end_sample - start_sample) <= 0
                    else chunk_wav[: (end_sample - start_sample)],
                    dtype=np.float32,
                )
                if len(chunk_wav) < (end_sample - start_sample):
                    pad_len = (end_sample - start_sample) - len(chunk_wav)
                    chunk_wav = np.pad(chunk_wav, (0, pad_len), mode="constant").astype(np.float32)
            chunks.append(AudioSegmentChunk(index=idx, start_sample=start_sample, end_sample=end_sample, wav=chunk_wav))

        if not chunks:
            chunks = [
                AudioSegmentChunk(
                    index=0,
                    start_sample=0,
                    end_sample=total_samples,
                    wav=np.asarray(wav, dtype=np.float32).reshape(-1),
                )
            ]
        return chunks

    def _enforce_max_duration(self, chunks: List[AudioSegmentChunk]) -> List[AudioSegmentChunk]:
        max_samples = int(self.vad_max_segment_s * WAV_SAMPLE_RATE)
        output: List[AudioSegmentChunk] = []
        for chunk in chunks:
            length = chunk.end_sample - chunk.start_sample
            if length <= max_samples:
                output.append(chunk)
                continue

            start = chunk.start_sample
            while start < chunk.end_sample:
                end = min(start + max_samples, chunk.end_sample)
                sub = chunk.wav[(start - chunk.start_sample) : (end - chunk.start_sample)]
                output.append(
                    AudioSegmentChunk(
                        index=len(output),
                        start_sample=start,
                        end_sample=end,
                        wav=np.asarray(sub, dtype=np.float32),
                    )
                )
                start = end

        for idx, chunk in enumerate(output):
            chunk.index = idx
        return output

    def _split_by_fixed_max(self, wav: np.ndarray) -> List[AudioSegmentChunk]:
        max_samples = int(self.vad_max_segment_s * WAV_SAMPLE_RATE)
        chunks = []
        for idx, start in enumerate(range(0, len(wav), max_samples)):
            end = min(start + max_samples, len(wav))
            chunks.append(
                AudioSegmentChunk(
                    index=idx,
                    start_sample=start,
                    end_sample=end,
                    wav=np.asarray(wav[start:end], dtype=np.float32),
                )
            )
        return chunks

    def _build_forced_aligner_metadata(
        self,
        requested: bool,
        language: str,
        text: str,
        segments: Optional[List[Dict]] = None,
        chunks: Optional[List[AudioSegmentChunk]] = None,
    ) -> Dict:
        if not requested:
            return {
                "requested": False,
                "available": False,
                "message": "Forced aligner not requested.",
                "language": language,
                "items": [],
                "text": text,
            }

        if not self.aligner_base_url:
            return {
                "requested": True,
                "available": False,
                "message": "Qwen3-ForcedAligner-0.6B is not deployed yet.",
                "language": language,
                "items": [],
                "text": text,
            }

        if not self.aligner_model:
            return {
                "requested": True,
                "available": False,
                "message": "Forced aligner model name is missing.",
                "language": language,
                "items": [],
                "text": text,
            }

        if not self.aligner_api_key:
            return {
                "requested": True,
                "available": False,
                "message": "Forced aligner API key is missing.",
                "language": language,
                "items": [],
                "text": text,
            }

        try:
            if self._aligner_client is None:
                self._aligner_client = RemoteForcedAlignerClient(
                    base_url=self.aligner_base_url,
                    api_key=self.aligner_api_key,
                    model=self.aligner_model,
                    timeout_s=self.aligner_timeout_s,
                    timestamp_segment_time_ms=self.aligner_timestamp_segment_time_ms,
                )

            items = []
            segments = segments or []
            chunks = chunks or []
            for segment, chunk in zip(segments, chunks):
                seg_text = (segment.get("text") or "").strip()
                if not seg_text:
                    continue
                seg_language = (segment.get("language") or language or "English").split(",")[0]
                seg_items = self._aligner_client.align_waveform(
                    wav=chunk.wav,
                    text=seg_text,
                    language=seg_language,
                )
                offset = float(segment.get("start_sec", 0.0))
                for item in seg_items:
                    items.append(
                        {
                            "text": item["text"],
                            "start_time": round(item["start_time"] + offset, 3),
                            "end_time": round(item["end_time"] + offset, 3),
                            "segment_index": int(segment.get("index", 0)),
                        }
                    )

            return {
                "requested": True,
                "available": True,
                "message": "Forced alignment succeeded.",
                "language": language,
                "items": items,
                "text": text,
            }
        except Exception as exc:
            return {
                "requested": True,
                "available": False,
                "message": f"Forced alignment failed: {exc}",
                "language": language,
                "items": [],
                "text": text,
            }
