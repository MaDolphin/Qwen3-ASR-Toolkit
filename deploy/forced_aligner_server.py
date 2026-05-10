"""Qwen3-ForcedAligner vLLM OpenAI server wrapper.

The local Qwen3 ASR package registers custom qwen3_asr config/model classes.
This wrapper imports and registers them before delegating to the vLLM OpenAI
server, so Qwen3-ForcedAligner can be served through /tokenize and /pooling.
"""

import uvloop
from vllm.entrypoints.cli.types import FlexibleArgumentParser
from vllm.entrypoints.openai.api_server import cli_env_setup, make_arg_parser, run_server, validate_parsed_serve_args
from vllm import ModelRegistry

import qwen_asr.inference.qwen3_asr  # noqa: F401
import qwen_asr.vllm_plugin as qwen3_vllm_plugin

FORCED_ALIGNER_ARCH = "Qwen3ASRForcedAlignerForTokenClassification"
FORCED_ALIGNER_TARGET = "qwen_asr.core.vllm_backend.qwen3_asr:Qwen3ASRForcedAlignerForTokenClassification"


def main() -> None:
    qwen3_vllm_plugin.register_qwen3_asr_model()
    ModelRegistry.register_model(FORCED_ALIGNER_ARCH, FORCED_ALIGNER_TARGET)
    cli_env_setup()
    parser = FlexibleArgumentParser(description="Qwen3-ForcedAligner vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
