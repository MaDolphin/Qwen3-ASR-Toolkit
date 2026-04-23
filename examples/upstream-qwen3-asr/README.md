# Upstream Qwen3-ASR Examples Snapshot

These files were moved out from `references/Qwen3-ASR-main/examples` so this project remains self-contained even after `references/` is removed.

Included files:

- `example_qwen3_asr_vllm.py`
- `example_qwen3_asr_vllm_streaming.py`
- `example_qwen3_asr_transformers.py`
- `example_qwen3_forced_aligner.py`
- `LICENSE.Qwen3-ASR` (upstream Apache-2.0 license)

Upstream source: QwenLM/Qwen3-ASR

## Running Notes

- These examples import the embedded local package: `qwen_asr`.
- Run from repository root, for example:
  - `python examples/upstream-qwen3-asr/example_qwen3_asr_vllm.py`
- Required dependencies are not part of the minimal toolkit install.
  - Install with extras:
    - `pip install -e ".[qwen-asr]"`
    - or for vLLM examples: `pip install -e ".[qwen-asr-vllm]"`
