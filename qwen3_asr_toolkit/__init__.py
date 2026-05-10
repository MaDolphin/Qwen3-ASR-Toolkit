"""Qwen3-ASR Toolkit 生产精简版。

该包只保留 Unified Native ASR Server 所需的工具模块：
音频加载、离线长音频切分转写、远端 forced aligner 客户端。
"""

from qwen3_asr_toolkit.offline_transcriber import OfflineTranscriber

__all__ = ["OfflineTranscriber"]
