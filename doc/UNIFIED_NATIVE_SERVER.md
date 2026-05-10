# Unified Native ASR Server

This deployment mode serves offline HTTP transcription and native WebSocket
streaming from one process and one loaded `Qwen3-ASR-1.7B` model instance.

## Why

Do not run both of these for the same GPU/model:

```bash
vllm serve models/Qwen3-ASR-1.7B --port 10010
python deploy/vllm_streaming_server_native.py --model-path models/Qwen3-ASR-1.7B
```

That loads `Qwen3-ASR-1.7B` twice. The unified native server loads it once with
`Qwen3ASRModel.LLM()` and shares the instance across:

- `POST /api/v1/offline/transcribe`
- `WS /ws/stream`
- `GET /health`

`Qwen3-ForcedAligner-0.6B` is a separate model. It is disabled by default and is
only used when forced alignment is explicitly requested and configured.

## Start

```bash
cd /workspace/project/Qwen3-ASR-Toolkit
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen-asr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN3_ASR_MODEL_PATH=/workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ASR-1.7B

python deploy/vllm_streaming_server_native.py \
  --host 0.0.0.0 \
  --port 10012 \
  --model-path /workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.8 \
  --max-new-tokens 128 \
  --chunk-size-sec 1.0 \
  --unfixed-chunk-num 2 \
  --unfixed-token-num 5 \
  --audio-queue-size 8 \
  --send-queue-size 32 \
  --decode-timeout-sec 0 \
  --enable-offline-api \
  --offline-num-threads 1 \
  --vad-target-segment-s 45 \
  --vad-max-segment-s 60 \
  --max-concurrent-asr-jobs 1 \
  --aligner-mode disabled
```

Use an absolute `--model-path` to avoid vLLM/Hugging Face treating a relative
path as a remote repo id.

## Health

```bash
curl http://127.0.0.1:10012/health
```

Expected capabilities include:

```json
{
  "offline_http": true,
  "native_websocket": true,
  "forced_aligner": "disabled",
  "max_concurrent_asr_jobs": 1,
  "asr_model_loaded_once": true
}
```

## Offline HTTP

```bash
curl -X POST "http://127.0.0.1:10012/api/v1/offline/transcribe" \
  -F "audio_file=@sample/sample_0.mp3" \
  -F "context=" \
  -F "use_forced_aligner=false"
```

The endpoint reuses `OfflineTranscriber` for audio loading, VAD segmentation,
`<=60s` segment enforcement, and response shape. It does not call
`OPENAI_BASE_URL`; it calls the shared native `Qwen3ASRModel` instance.

## Native WebSocket

Use the existing native protocol:

```text
WS /ws/stream
```

For one connection, keep audio at or below 120 seconds. Longer realtime audio
should use the windowed client.

## Long Audio Realtime

Use service-layer rolling windows:

```bash
python deploy/test_native_streaming_windowed_harness.py \
  --uri ws://127.0.0.1:10012/ws/stream \
  --input sample/sample_2.m4a \
  --reference sample/sample_2.txt \
  --output runtime/native_streaming_validation/unified_sample_2_windowed.json \
  --report runtime/native_streaming_validation/UNIFIED_SAMPLE_2_WINDOWED_REPORT.md \
  --case-dir runtime/native_streaming_validation/unified_sample_2_windowed_cases \
  --case-label unified_sample_2_windowed \
  --window-sec 120 \
  --overlap-sec 0 \
  --chunk-ms 500
```

This validates a service-layer workaround. It is not KV-cache true incremental
streaming because each native session still uses the current cumulative
`generate()` implementation.

## Forced Aligner

Default:

```bash
--aligner-mode disabled
```

When `use_forced_aligner=true`, the server returns forced-aligner metadata that
states alignment is disabled instead of loading `Qwen3-ForcedAligner-0.6B`.

Remote mode:

```bash
--aligner-mode remote \
--aligner-base-url http://127.0.0.1:10013 \
--aligner-api-key EMPTY \
--aligner-model-path Qwen3-ForcedAligner-0.6B
```

`local-lazy` is reserved in the CLI but intentionally returns an explicit error
in this server until GPU memory and lifecycle behavior are validated.

## Concurrency

The server shares one model instance across offline HTTP and WebSocket decode
calls. Access is guarded by `--max-concurrent-asr-jobs`, default `1`, to avoid
HTTP jobs and realtime sessions competing unpredictably for the same vLLM model.

Recommended first deployment:

```text
offline_num_threads=1
max_concurrent_asr_jobs=1
```

Increase only after GPU latency and memory are validated.
