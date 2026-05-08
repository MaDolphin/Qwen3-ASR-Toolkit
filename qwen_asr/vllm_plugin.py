# coding=utf-8
"""vLLM plugin hooks for embedded Qwen3-ASR support."""

from vllm import ModelRegistry


MODEL_ARCH = "Qwen3ASRForConditionalGeneration"
MODEL_TARGET = "qwen_asr.core.vllm_backend.qwen3_asr:Qwen3ASRForConditionalGeneration"


def register_qwen3_asr_model() -> None:
    ModelRegistry.register_model(MODEL_ARCH, MODEL_TARGET)
