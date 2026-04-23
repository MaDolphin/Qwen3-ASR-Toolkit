import os
import io
import time
import random
import base64

import re
import numpy as np
import soundfile as sf

from openai import OpenAI
from pydub import AudioSegment

from qwen3_asr_toolkit.env_utils import load_project_dotenv

MAX_API_RETRY = 10
API_RETRY_SLEEP = (1, 2)


language_code_mapping = {
    "ar": "Arabic",
    "zh": "Chinese",
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish"
}


class QwenASR:
    def __init__(self, model: str = "qwen3-asr-flash", base_url: str = None, api_key: str = None):
        load_project_dotenv()
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        )

    def post_text_process(self, text, threshold=20):
        def fix_char_repeats(s, thresh):
            res = []
            i = 0
            n = len(s)
            while i < n:
                count = 1
                while i + count < n and s[i + count] == s[i]:
                    count += 1

                if count > thresh:
                    res.append(s[i])
                    i += count
                else:
                    res.append(s[i:i + count])
                    i += count
            return ''.join(res)

        def fix_pattern_repeats(s, thresh, max_len=20):
            n = len(s)
            min_repeat_chars = thresh * 2
            if n < min_repeat_chars:
                return s

            i = 0
            result = []
            while i <= n - min_repeat_chars:
                found = False
                for k in range(1, max_len + 1):
                    if i + k * thresh > n:
                        break

                    pattern = s[i:i + k]

                    valid = True
                    for rep in range(1, thresh):
                        start_idx = i + rep * k
                        if s[start_idx:start_idx + k] != pattern:
                            valid = False
                            break

                    if valid:
                        total_rep = thresh
                        end_index = i + thresh * k
                        while end_index + k <= n and s[end_index:end_index + k] == pattern:
                            total_rep += 1
                            end_index += k

                        result.append(pattern)
                        result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                        i = n
                        found = True
                        break

                if found:
                    break
                else:
                    result.append(s[i])
                    i += 1

            if not found:
                result.append(s[i:])
            return ''.join(result)

        text = fix_char_repeats(text, threshold)
        return fix_pattern_repeats(text, threshold)

    def _encode_audio_base64(self, file_path: str) -> str:
        """Read an audio file and return its base64-encoded data URI."""
        ext = os.path.splitext(file_path)[1].lstrip('.').lower()
        mime_map = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'ogg': 'audio/ogg',
            'flac': 'audio/flac',
            'm4a': 'audio/mp4',
        }
        mime_type = mime_map.get(ext, f'audio/{ext}')
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return f"data:{mime_type};base64,{b64}"

    def _encode_audio_bytes_base64(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def _parse_asr_output(self, raw_text: str):
        language = "Not Supported"
        recog_text = raw_text or ""

        match = re.match(r'^language\s+(\w+)\s*<asr_text>(.*)', recog_text, re.DOTALL)
        if match:
            detected_lang = match.group(1)
            recog_text = match.group(2).strip()
            if detected_lang in language_code_mapping.values():
                language = detected_lang
            else:
                language = language_code_mapping.get(detected_lang, detected_lang)
        return language, self.post_text_process(recog_text)

    def _run_chat_completions(self, wav_url: str, context: str = ""):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": context},
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": wav_url},
                    },
                ]
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        choice = response.choices[0]
        raw_text = choice.message.content or ""
        return self._parse_asr_output(raw_text)

    def asr_waveform(self, wav: np.ndarray, sr: int = 16000, context: str = ""):
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        with io.BytesIO() as buff:
            sf.write(buff, wav, sr, format="WAV")
            audio_bytes = buff.getvalue()
        wav_url = self._encode_audio_bytes_base64(audio_bytes, "audio/wav")

        response = None
        for idx in range(MAX_API_RETRY):
            try:
                return self._run_chat_completions(wav_url=wav_url, context=context)
            except Exception as e:
                err_msg = str(e)
                print(f"Retry {idx + 1}... waveform\n{err_msg}")
                if "encoder cache size" in err_msg:
                    raise
                response = e
            time.sleep(random.uniform(*API_RETRY_SLEEP))
        raise Exception(f"ASR task failed after {MAX_API_RETRY} retries!\n{response}")

    def asr(self, wav_url: str, context: str = ""):
        if not wav_url.startswith("http"):
            assert os.path.exists(wav_url), f"{wav_url} not exists!"
            file_path = wav_url
            file_size = os.path.getsize(file_path)

            # file size > 10M
            if file_size > 10 * 1024 * 1024:
                # convert to mp3
                mp3_path = os.path.splitext(file_path)[0] + ".mp3"
                audio = AudioSegment.from_file(file_path)
                audio.export(mp3_path, format="mp3")
                file_path = mp3_path

            # Encode audio as base64 data URI for OpenAI-compatible API
            wav_url = self._encode_audio_base64(file_path)

        # Submit the ASR task
        response = None
        for idx in range(MAX_API_RETRY):
            try:
                return self._run_chat_completions(wav_url=wav_url, context=context)
            except Exception as e:
                err_msg = str(e)
                print(f"Retry {idx + 1}...  {wav_url[:100]}...\n{err_msg}")
                # Don't retry on encoder cache size exceeded - this won't resolve with retries
                if "encoder cache size" in err_msg:
                    raise
                response = e
            time.sleep(random.uniform(*API_RETRY_SLEEP))
        raise Exception(f"ASR task failed after {MAX_API_RETRY} retries!\n{response}")


if __name__ == "__main__":
    load_project_dotenv()
    qwen_asr = QwenASR(
        model=os.environ.get("QWEN3_ASR_MODEL", "qwen3-asr-flash"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    asr_text = qwen_asr.asr(wav_url="/path/to/your/wav_file.wav")
    print(asr_text)
