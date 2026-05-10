#!/usr/bin/env bash
set -euo pipefail

activate_conda_env() {
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    conda activate "$CONDA_ENV"
  elif [[ -d "/opt/conda/envs/$CONDA_ENV" ]]; then
    conda activate "/opt/conda/envs/$CONDA_ENV"
  else
    echo "Conda environment not found: $CONDA_ENV" >&2
    conda info --envs >&2 || true
    exit 1
  fi
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/deploy_native_asr.sh

下载/检查 Qwen3-ASR 与 Qwen3-ForcedAligner 模型，并以低显存默认参数启动：
  - ForcedAligner vLLM pooling server: 127.0.0.1:10013
  - Qwen3-ASR Native server: 0.0.0.0:10012

常用环境变量：
  CONDA_ENV=qwen-asr
  ASR_GPU_MEMORY_UTILIZATION=0.50
  ASR_KV_CACHE_MEMORY_BYTES=8G
  ALIGNER_GPU_MEMORY_UTILIZATION=0.10
  ALIGNER_KV_CACHE_MEMORY_BYTES=2G
EOF
  exit 0
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

CONDA_ENV="${CONDA_ENV:-qwen-asr}"
ASR_MODEL_DIR="${ASR_MODEL_DIR:-$PROJECT_DIR/models/Qwen3-ASR-1.7B}"
ALIGNER_MODEL_DIR="${ALIGNER_MODEL_DIR:-$PROJECT_DIR/models/Qwen3-ForcedAligner-0.6B}"
ASR_HOST="${ASR_HOST:-0.0.0.0}"
ASR_PORT="${ASR_PORT:-10012}"
ALIGNER_HOST="${ALIGNER_HOST:-127.0.0.1}"
ALIGNER_PORT="${ALIGNER_PORT:-10013}"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_DIR/runtime/native_deploy}"

ASR_GPU_MEMORY_UTILIZATION="${ASR_GPU_MEMORY_UTILIZATION:-0.50}"
ASR_KV_CACHE_MEMORY_BYTES="${ASR_KV_CACHE_MEMORY_BYTES:-8G}"
ASR_CPU_OFFLOAD_GB="${ASR_CPU_OFFLOAD_GB:-0}"
ASR_MAX_MODEL_LEN="${ASR_MAX_MODEL_LEN:-65536}"
ALIGNER_GPU_MEMORY_UTILIZATION="${ALIGNER_GPU_MEMORY_UTILIZATION:-0.10}"
ALIGNER_KV_CACHE_MEMORY_BYTES="${ALIGNER_KV_CACHE_MEMORY_BYTES:-2G}"
ALIGNER_CPU_OFFLOAD_GB="${ALIGNER_CPU_OFFLOAD_GB:-0}"

mkdir -p "$RUNTIME_DIR"

activate_conda_env

python -m pip install -U pip "setuptools<81.0.0,>=77.0.3" wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install modelscope

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export QWEN3_ASR_MODEL_PATH="$ASR_MODEL_DIR"

nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits > "$RUNTIME_DIR/gpu_before.txt" || true

if [[ ! -f "$ASR_MODEL_DIR/config.json" || ! -f "$ASR_MODEL_DIR/model.safetensors.index.json" ]]; then
  mkdir -p "$ASR_MODEL_DIR"
  modelscope download --model 'Qwen/Qwen3-ASR-1.7B' --local_dir "$ASR_MODEL_DIR"
fi

if [[ ! -f "$ALIGNER_MODEL_DIR/config.json" || ! -f "$ALIGNER_MODEL_DIR/model.safetensors" ]]; then
  mkdir -p "$ALIGNER_MODEL_DIR"
  modelscope download --model 'Qwen/Qwen3-ForcedAligner-0.6B' --local_dir "$ALIGNER_MODEL_DIR"
fi

if [[ -f "$RUNTIME_DIR/aligner_server.pid" ]] && kill -0 "$(cat "$RUNTIME_DIR/aligner_server.pid")" 2>/dev/null; then
  echo "ForcedAligner already running: $(cat "$RUNTIME_DIR/aligner_server.pid")"
else
  setsid python -m deploy.forced_aligner_server \
    --host "$ALIGNER_HOST" \
    --port "$ALIGNER_PORT" \
    --model "$ALIGNER_MODEL_DIR" \
    --served-model-name Qwen3-ForcedAligner-0.6B \
    --runner pooling \
    --pooler-config '{"task":"token_classify"}' \
    --hf-overrides '{"architectures":["Qwen3ASRForcedAlignerForTokenClassification"]}' \
    --gpu-memory-utilization "$ALIGNER_GPU_MEMORY_UTILIZATION" \
    --kv-cache-memory-bytes "$ALIGNER_KV_CACHE_MEMORY_BYTES" \
    --cpu-offload-gb "$ALIGNER_CPU_OFFLOAD_GB" \
    --disable-log-requests \
    > "$RUNTIME_DIR/aligner_server.log" 2>&1 &
  echo $! > "$RUNTIME_DIR/aligner_server.pid"
fi

for _ in $(seq 1 180); do
  if curl -fsS "http://$ALIGNER_HOST:$ALIGNER_PORT/health" > "$RUNTIME_DIR/health_aligner.json" 2>/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS "http://$ALIGNER_HOST:$ALIGNER_PORT/health" > "$RUNTIME_DIR/health_aligner.json" || {
  tail -100 "$RUNTIME_DIR/aligner_server.log" >&2 || true
  exit 1
}
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits > "$RUNTIME_DIR/gpu_after_aligner.txt" || true

if [[ -f "$RUNTIME_DIR/asr_server.pid" ]] && kill -0 "$(cat "$RUNTIME_DIR/asr_server.pid")" 2>/dev/null; then
  echo "ASR server already running: $(cat "$RUNTIME_DIR/asr_server.pid")"
else
  setsid qwen3-asr-native-server \
    --host "$ASR_HOST" \
    --port "$ASR_PORT" \
    --model-path "$ASR_MODEL_DIR" \
    --gpu-memory-utilization "$ASR_GPU_MEMORY_UTILIZATION" \
    --kv-cache-memory-bytes "$ASR_KV_CACHE_MEMORY_BYTES" \
    --cpu-offload-gb "$ASR_CPU_OFFLOAD_GB" \
    --max-model-len "$ASR_MAX_MODEL_LEN" \
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
    --aligner-mode remote \
    --aligner-base-url "http://$ALIGNER_HOST:$ALIGNER_PORT" \
    --aligner-api-key EMPTY \
    --aligner-model-path Qwen3-ForcedAligner-0.6B \
    > "$RUNTIME_DIR/asr_server.log" 2>&1 &
  echo $! > "$RUNTIME_DIR/asr_server.pid"
fi

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$ASR_PORT/health" > "$RUNTIME_DIR/health_asr.json" 2>/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:$ASR_PORT/health" > "$RUNTIME_DIR/health_asr.json" || {
  tail -100 "$RUNTIME_DIR/asr_server.log" >&2 || true
  bash scripts/stop_native_asr.sh || true
  exit 1
}
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits > "$RUNTIME_DIR/gpu_after_asr.txt" || true

echo "部署完成："
echo "  ASR:     http://127.0.0.1:$ASR_PORT"
echo "  WS:      ws://127.0.0.1:$ASR_PORT/ws/stream"
echo "  Aligner: http://$ALIGNER_HOST:$ALIGNER_PORT"
echo "  Runtime: $RUNTIME_DIR"
