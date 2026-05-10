#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen-asr

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export QWEN3_ASR_MODEL_PATH="${QWEN3_ASR_MODEL_PATH:-$PWD/models/Qwen3-ASR-1.7B}"

qwen3-asr-native-server \
  --host "${QWEN3_ASR_HOST:-0.0.0.0}" \
  --port "${QWEN3_ASR_PORT:-10012}" \
  --model-path "$QWEN3_ASR_MODEL_PATH" \
  --gpu-memory-utilization "${QWEN3_ASR_GPU_MEMORY_UTILIZATION:-0.8}" \
  --max-new-tokens "${QWEN3_ASR_MAX_NEW_TOKENS:-128}" \
  --chunk-size-sec "${QWEN3_ASR_CHUNK_SIZE_SEC:-1.0}" \
  --unfixed-chunk-num "${QWEN3_ASR_UNFIXED_CHUNK_NUM:-2}" \
  --unfixed-token-num "${QWEN3_ASR_UNFIXED_TOKEN_NUM:-5}" \
  --audio-queue-size "${QWEN3_ASR_AUDIO_QUEUE_SIZE:-8}" \
  --send-queue-size "${QWEN3_ASR_SEND_QUEUE_SIZE:-32}" \
  --decode-timeout-sec "${QWEN3_ASR_DECODE_TIMEOUT_SEC:-0}" \
  --enable-offline-api \
  --offline-num-threads "${QWEN3_ASR_OFFLINE_NUM_THREADS:-1}" \
  --vad-target-segment-s "${QWEN3_ASR_VAD_TARGET_SEGMENT_S:-45}" \
  --vad-max-segment-s "${QWEN3_ASR_VAD_MAX_SEGMENT_S:-60}" \
  --max-concurrent-asr-jobs "${QWEN3_ASR_MAX_CONCURRENT_JOBS:-1}" \
  --aligner-mode "${QWEN3_ALIGNER_MODE:-disabled}"
