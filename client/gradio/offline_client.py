from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


class OfflineTranscribeError(RuntimeError):
    pass


def transcribe_offline(
    api_url: str,
    audio_path: str,
    context: str = "",
    use_forced_aligner: bool = False,
    timeout_sec: float = 1800.0,
) -> dict[str, Any]:
    input_path = Path(audio_path)
    if not input_path.exists():
        raise OfflineTranscribeError(f"音频文件不存在：{input_path}")

    try:
        with input_path.open("rb") as file_obj:
            files = {"audio_file": (input_path.name, file_obj, "application/octet-stream")}
            data = {
                "context": context or "",
                "use_forced_aligner": str(bool(use_forced_aligner)).lower(),
            }
            response = requests.post(api_url, files=files, data=data, timeout=timeout_sec)
            response.raise_for_status()
            return response.json()
    except requests.RequestException as exc:
        raise OfflineTranscribeError(f"离线转写请求失败：{exc}") from exc
