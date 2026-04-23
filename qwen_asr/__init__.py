# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
qwen_asr: embedded upstream Qwen3-ASR package.
"""

from qwen_asr.inference.utils import parse_asr_output

__version__ = "0.0.6-embedded"

_DEPENDENCY_HINT = (
    "Failed to import qwen_asr model classes. "
    "Please install Qwen-ASR dependencies (for example: "
    "`pip install \"transformers==4.57.6\" \"nagisa==0.2.11\" "
    "\"soynlp==0.0.493\" \"accelerate==1.12.0\" qwen-omni-utils sox pytz`)."
)


def __getattr__(name):
    if name == "Qwen3ASRModel":
        try:
            from qwen_asr.inference.qwen3_asr import Qwen3ASRModel as _Qwen3ASRModel
            return _Qwen3ASRModel
        except Exception as exc:
            raise ImportError(_DEPENDENCY_HINT) from exc
    if name == "Qwen3ForcedAligner":
        try:
            from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner as _Qwen3ForcedAligner
            return _Qwen3ForcedAligner
        except Exception as exc:
            raise ImportError(_DEPENDENCY_HINT) from exc
    raise AttributeError(f"module 'qwen_asr' has no attribute '{name}'")


__all__ = [
    "__version__",
    "Qwen3ASRModel",
    "Qwen3ForcedAligner",
    "parse_asr_output",
]
