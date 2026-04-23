import io
import base64
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import soundfile as sf

from qwen3_asr_toolkit.audio_tools import WAV_SAMPLE_RATE


class ForceAlignTextProcessor:
    def is_kept_char(self, ch: str) -> bool:
        if ch == "'":
            return True
        category = unicodedata.category(ch)
        if category.startswith("L") or category.startswith("N"):
            return True
        return False

    def clean_token(self, token: str) -> str:
        return "".join(ch for ch in token if self.is_kept_char(ch))

    def is_cjk_char(self, ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF
        )

    def split_segment_with_chinese(self, seg: str) -> List[str]:
        tokens: List[str] = []
        buffer: List[str] = []

        def flush_buffer():
            nonlocal buffer
            if buffer:
                tokens.append("".join(buffer))
                buffer = []

        for ch in seg:
            if self.is_cjk_char(ch):
                flush_buffer()
                tokens.append(ch)
            else:
                buffer.append(ch)

        flush_buffer()
        return tokens

    def tokenize_space_lang(self, text: str) -> List[str]:
        tokens: List[str] = []
        for seg in text.split():
            cleaned = self.clean_token(seg)
            if cleaned:
                tokens.extend(self.split_segment_with_chinese(cleaned))
        return tokens

    def encode_timestamp(self, text: str, language: str) -> Tuple[List[str], str]:
        # For server-side forced-aligner pooling API, we keep robust tokenization:
        # CJK chars are split by char, space languages by words.
        word_list = self.tokenize_space_lang(text)
        if not word_list:
            return [], ""
        input_text = "<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"
        input_text = "<|audio_start|><|audio_pad|><|audio_end|>" + input_text
        return word_list, input_text

    def fix_timestamp(self, data: List[int]) -> List[int]:
        values = list(data)
        n = len(values)
        if n <= 1:
            return values

        dp = [1] * n
        parent = [-1] * n
        for i in range(1, n):
            for j in range(i):
                if values[j] <= values[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

        max_length = max(dp)
        max_idx = dp.index(max_length)
        lis_indices = []
        idx = max_idx
        while idx != -1:
            lis_indices.append(idx)
            idx = parent[idx]
        lis_indices.reverse()

        is_normal = [False] * n
        for i in lis_indices:
            is_normal[i] = True

        result = values.copy()
        i = 0
        while i < n:
            if is_normal[i]:
                i += 1
                continue

            j = i
            while j < n and not is_normal[j]:
                j += 1

            anomaly_count = j - i
            left_val = None
            right_val = None
            for k in range(i - 1, -1, -1):
                if is_normal[k]:
                    left_val = result[k]
                    break
            for k in range(j, n):
                if is_normal[k]:
                    right_val = result[k]
                    break

            if anomaly_count <= 2:
                for k in range(i, j):
                    if left_val is None and right_val is not None:
                        result[k] = right_val
                    elif right_val is None and left_val is not None:
                        result[k] = left_val
                    elif left_val is not None and right_val is not None:
                        result[k] = left_val if (k - (i - 1)) <= (j - k) else right_val
            else:
                if left_val is not None and right_val is not None:
                    step = (right_val - left_val) / (anomaly_count + 1)
                    for k in range(i, j):
                        result[k] = left_val + step * (k - i + 1)
                elif left_val is not None:
                    for k in range(i, j):
                        result[k] = left_val
                elif right_val is not None:
                    for k in range(i, j):
                        result[k] = right_val
            i = j

        return [int(x) for x in result]

    def parse_timestamp(self, word_list: List[str], timestamp_ms: List[int]) -> List[Dict]:
        fixed = self.fix_timestamp(timestamp_ms)
        output = []
        for i, word in enumerate(word_list):
            start_time = fixed[i * 2]
            end_time = fixed[i * 2 + 1]
            output.append({"text": word, "start_time_ms": start_time, "end_time_ms": end_time})
        return output


class RemoteForcedAlignerClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: int = 120,
        timestamp_segment_time_ms: int = 80,
        timestamp_token_id: Optional[int] = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = int(timeout_s)
        self.timestamp_segment_time_ms = int(timestamp_segment_time_ms)
        self.timestamp_token_id = timestamp_token_id
        self.processor = ForceAlignTextProcessor()

    def align_waveform(self, wav: np.ndarray, text: str, language: str) -> List[Dict]:
        text = (text or "").strip()
        if not text:
            return []

        word_list, input_text = self.processor.encode_timestamp(text, language)
        if not word_list or not input_text:
            return []

        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        duration_sec = len(wav) / WAV_SAMPLE_RATE if len(wav) > 0 else 0.0
        audio_data_url = self._wav_to_data_url(wav)

        input_ids = self._tokenize(input_text)
        if self.timestamp_token_id is None:
            self.timestamp_token_id = self._get_timestamp_token_id()

        logits = self._pool_token_classify(input_text=input_text, audio_data_url=audio_data_url)
        output_ids = np.asarray(logits, dtype=np.float32).argmax(axis=1).tolist()

        timestamp_class_ids = [out_id for token_id, out_id in zip(input_ids, output_ids) if token_id == self.timestamp_token_id]
        timestamp_ms = self._normalize_timestamp_count(timestamp_class_ids, len(word_list), duration_sec)

        aligned = self.processor.parse_timestamp(word_list, timestamp_ms)
        output = []
        for item in aligned:
            start_sec = max(0.0, min(duration_sec, item["start_time_ms"] / 1000.0))
            end_sec = max(start_sec, min(duration_sec, item["end_time_ms"] / 1000.0))
            output.append(
                {
                    "text": item["text"],
                    "start_time": round(start_sec, 3),
                    "end_time": round(end_sec, 3),
                }
            )
        return output

    def _normalize_timestamp_count(self, class_ids: List[int], word_count: int, duration_sec: float) -> List[int]:
        expected = word_count * 2
        ms_values = [int(x * self.timestamp_segment_time_ms) for x in class_ids]

        if expected <= 0:
            return []

        if len(ms_values) >= expected:
            return ms_values[:expected]

        if len(ms_values) == 0:
            max_ms = int(duration_sec * 1000.0)
            if max_ms <= 0:
                return [0] * expected
            return [int(i * max_ms / max(1, expected - 1)) for i in range(expected)]

        last = ms_values[-1]
        while len(ms_values) < expected:
            ms_values.append(last)
        return ms_values

    def _tokenize(self, prompt: str) -> List[int]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "add_special_tokens": False,
        }
        response = self._post_json(["/tokenize", "/v1/tokenize"], payload)
        return response.get("tokens", [])

    def _get_timestamp_token_id(self) -> int:
        token_ids = self._tokenize("<timestamp>")
        if not token_ids:
            raise RuntimeError("Unable to retrieve timestamp token id from forced aligner server.")
        return int(token_ids[0])

    def _pool_token_classify(self, input_text: str, audio_data_url: str) -> List[List[float]]:
        payload = {
            "model": self.model,
            "task": "token_classify",
            "input": input_text,
            "add_special_tokens": False,
            "multi_modal_data": {
                "audio": [audio_data_url],
            },
        }
        response = self._post_json(["/pooling", "/v1/pooling"], payload)
        data = response.get("data", [])
        if not data:
            raise RuntimeError(f"Unexpected pooling response: {response}")
        rows = data[0].get("data", [])
        if not rows:
            raise RuntimeError(f"Empty token_classify result: {response}")
        return rows

    def _post_json(self, paths: List[str], payload: Dict) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = None
        for path in paths:
            url = f"{self.base_url}{path}"
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            if response.status_code == 404:
                last_error = RuntimeError(f"Endpoint not found: {url}")
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"Forced aligner request failed ({response.status_code}): {response.text}")
            return response.json()
        raise last_error or RuntimeError("No valid forced aligner endpoint found.")

    def _wav_to_data_url(self, wav: np.ndarray) -> str:
        with io.BytesIO() as buff:
            sf.write(buff, wav, WAV_SAMPLE_RATE, format="WAV")
            wav_bytes = buff.getvalue()
        b64 = base64.b64encode(wav_bytes).decode("utf-8")
        return f"data:audio/wav;base64,{b64}"
