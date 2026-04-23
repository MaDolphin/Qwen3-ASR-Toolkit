# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-04-23

### Breaking Changes
- **API Backend**: Migrated from Alibaba Cloud DashScope to self-hosted vLLM with OpenAI-compatible endpoints.
- **SDK**: Replaced `dashscope` with `openai` SDK.
- **Authentication**: Switched from `DASHSCOPE_API_KEY` to `OPENAI_API_KEY` + `OPENAI_BASE_URL`.
- **Audio Transport**: Changed from `file://` local path to base64-encoded Data URI.
- **Response Parsing**: Language and text are now embedded in model output (`language XXX<asr_text>...`) instead of structured annotations.

### Added
- **Server**: New FastAPI + Uvicorn server (`qwen3_asr_toolkit.server`) providing:
  - Offline transcription REST API: `POST /api/v1/offline/transcribe`
  - Real-time streaming WebSocket API: `/ws/v1/realtime/transcribe`
- **Offline Transcriber**: `OfflineTranscriber` with VAD-based chunking, strict 60s max segment enforcement, and fallback splitting.
- **Streaming Transcriber**: `StreamingTranscriber` + `StreamingSession` for quasi-realtime incremental transcription with silence-based segment finalization.
- **Forced Aligner Client**: `RemoteForcedAlignerClient` to integrate with `Qwen3-ForcedAligner-0.6B` via `/tokenize` and `/pooling` endpoints.
- **CLI Tools**:
  - `qwen3-asr-offline-cli`: REST-based offline transcription client
  - `qwen3-asr-stream-cli`: WebSocket-based real-time streaming client
  - `qwen3-asr-server`: Server entry point
- **Embedded Upstream Package**: Migrated `qwen_asr/` into the main repository (previously in `references/`), including transformers and vLLM backends.
- **Environment Configuration**: Unified `.env` configuration via `env_utils.py` with `.env.example` template.
- **Documentation Suite**:
  - `doc/DEPLOYMENT.md`: Deployment guide
  - `doc/USAGE.md`: Usage manual for REST, WebSocket, and CLI
  - `doc/DEVELOPER_GUIDE.md`: Developer guide
  - `doc/TECH_IMPROVEMENTS.md`: Technical improvements and known issues
  - `doc/cloud_api_ws_usage.md`: Cloud API and WebSocket protocol reference
  - `doc/vllm_migration.md`: DashScope to vLLM migration log
- **Tests**:
  - `tests/test_offline_and_streaming.py`: Unit tests for chunking constraints, VAD fallback, and streaming state machine
  - `tests/test_server_api.py`: REST and WebSocket protocol tests
  - `tests/test_qwen_asr_embedding.py`: Import and parsing tests for embedded upstream package
- **Deployment Artifacts**:
  - `deploy/systemd/qwen3-asr-toolkit.service`: systemd service template
  - `scripts/deploy_server.sh`: Server deployment helper script
  - `scripts/run_sample_tests.sh`: Sample verification script
- **Samples**: Added `sample/deutsch.mp3`, `sample/sample_2.m4a`, and corresponding ground-truth `.txt` files.

### Changed
- `setup.py`: Version bumped to `2.0.0`; added `fastapi`, `uvicorn`, `websockets`, `python-multipart`, `python-dotenv` to dependencies; added extras for `qwen-asr` and `qwen-asr-vllm`.
- `requirements.txt`: Replaced `dashscope` with `openai`; added server dependencies.
- `qwen3asr.py`: Rewritten to use OpenAI-compatible client with base64 audio encoding.
- `call_api.py`: Updated arguments for vLLM endpoints; added model auto-detection.

### Removed
- `dashscope` dependency and all DashScope-specific API calls.
- External `references/` directory (replaced by embedded `qwen_asr/`).

---

## [1.0.4] - 2026-04-16

### Added
- PyPI package distribution as `qwen3-asr-toolkit`.
- Long audio support via VAD-based splitting (default target 120s).
- Multi-threaded parallel API calls.
- SRT subtitle generation.
- Universal media support via FFmpeg.

### Notes
- This was the final DashScope-based release before the v2.0.0 rewrite.

---

## [1.0.3] - 2026-04-15

### Added
- Additional CLI options and post-processing for hallucination removal.

## [1.0.2] - 2026-04-15

### Added
- Improved audio resampling and error handling.

## [1.0.1] - 2026-04-14

### Added
- Initial public release with core offline transcription CLI.

## [1.0.0] - 2026-04-14

### Added
- Initial commit.
