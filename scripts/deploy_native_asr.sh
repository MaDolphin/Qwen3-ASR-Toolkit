#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/deploy_native_asr.sh [options]

Options:
  --env-file PATH          读取指定 env 文件，默认 .env
  --no-conda-activate      不执行 conda activate，即使 env 文件中设置了 CONDA_ENV
  -h, --help               显示帮助

说明：
  部署配置强制来自 env 文件。默认 .env 不存在时会从 .env.example 自动创建。
  请在 .env 中配置 GPU、模型路径、端口、显存、并发和 Gradio 参数。
  功能测试完成后如需自动停止服务：STOP_AFTER_TEST=1 bash scripts/test_native_asr_functional.sh
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE=".env"
NO_CONDA_ACTIVATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="${2:-}"
      if [[ -z "$ENV_FILE" ]]; then
        echo "Missing value for --env-file" >&2
        exit 2
      fi
      shift 2
      ;;
    --no-conda-activate)
      NO_CONDA_ACTIVATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

prepare_env_file() {
  local env_file="$1"
  if [[ -f "$env_file" ]]; then
    return 0
  fi
  if [[ "$env_file" == ".env" || "$env_file" == "$SCRIPT_DIR/.env" ]]; then
    cp .env.example "$env_file"
    echo "已从 .env.example 创建 $env_file，请按需修改 GPU、端口和模型路径。"
    return 0
  fi
  echo "Env file not found: $env_file" >&2
  exit 1
}

load_env_file() {
  local env_file="$1"
  prepare_env_file "$env_file"
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
}

activate_conda_env() {
  local env_name="${1:-}"
  if [[ -z "$env_name" ]]; then
    echo "Using current Python environment: $(command -v python)"
    return 0
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    conda activate "$env_name"
  elif [[ -d "/opt/conda/envs/$env_name" ]]; then
    conda activate "/opt/conda/envs/$env_name"
  else
    echo "Conda environment not found: $env_name" >&2
    conda info --envs >&2 || true
    exit 1
  fi
}

wait_for_http() {
  local url="$1"
  local output="$2"
  local log_file="$3"
  local timeout_sec="${4:-360}"
  local insecure="${5:-0}"
  local tries=$((timeout_sec / 2))
  local curl_args=(-fsS)
  if [[ "$insecure" == "1" ]]; then
    curl_args+=(-k)
  fi
  [[ "$tries" -lt 1 ]] && tries=1
  for _ in $(seq 1 "$tries"); do
    if curl "${curl_args[@]}" "$url" > "$output" 2>/dev/null; then
      return 0
    fi
    sleep 2
  done
  curl "${curl_args[@]}" "$url" > "$output" || {
    tail -100 "$log_file" >&2 || true
    return 1
  }
}

print_tcp_port_listeners() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | awk -v port=":$port" '$4 ~ port {print}' >&2 || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >&2 || true
  fi
}

ensure_tcp_port_available() {
  local port="$1"
  local name="$2"
  if command -v ss >/dev/null 2>&1; then
    if ss -ltn 2>/dev/null | awk -v port=":$port" '$4 ~ port {found=1} END {exit found ? 0 : 1}'; then
      echo "$name port $port is already in use." >&2
      print_tcp_port_listeners "$port"
      return 1
    fi
  elif command -v lsof >/dev/null 2>&1; then
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "$name port $port is already in use." >&2
      print_tcp_port_listeners "$port"
      return 1
    fi
  fi
}

start_process_if_needed() {
  local pid_file="$1"
  local name="$2"
  shift 2
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name already running: $(cat "$pid_file")"
    return 0
  fi
  "$@" &
  echo $! > "$pid_file"
}

load_env_file "$ENV_FILE"

required_vars=(
  PROJECT_DIR
  QWEN3_CUDA_VISIBLE_DEVICES
  QWEN3_ASR_MODEL_PATH
  QWEN3_ALIGNER_MODEL_PATH
  QWEN3_ASR_PORT
  QWEN3_ALIGNER_PORT
  QWEN3_GRADIO_PORT
  QWEN3_ASR_MAX_CONCURRENT_JOBS
  QWEN3_ASR_OFFLINE_NUM_THREADS
)
missing_vars=()
for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    missing_vars+=("$var_name")
  fi
done
if [[ "${#missing_vars[@]}" -gt 0 ]]; then
  echo "Env file $ENV_FILE is missing required variables:" >&2
  printf '  - %s\n' "${missing_vars[@]}" >&2
  echo "请参考 .env.example 整理 $ENV_FILE。" >&2
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
cd "$PROJECT_DIR"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_DIR/runtime/native_deploy}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
mkdir -p "$RUNTIME_DIR" "$LOG_DIR"

if [[ "$NO_CONDA_ACTIVATE" != "1" ]]; then
  activate_conda_env "${CONDA_ENV:-}"
else
  echo "Skip conda activation. Using current Python environment: $(command -v python)"
fi

ASR_MODEL_DIR="${QWEN3_ASR_MODEL_PATH:?QWEN3_ASR_MODEL_PATH is required}"
ALIGNER_MODEL_DIR="${QWEN3_ALIGNER_MODEL_PATH:?QWEN3_ALIGNER_MODEL_PATH is required}"
ASR_HOST="${QWEN3_ASR_HOST:-0.0.0.0}"
ASR_PORT="${QWEN3_ASR_PORT:-10012}"
ALIGNER_HOST="${QWEN3_ALIGNER_HOST:-0.0.0.0}"
ALIGNER_PORT="${QWEN3_ALIGNER_PORT:-10013}"
GRADIO_HOST="${QWEN3_GRADIO_HOST:-0.0.0.0}"
GRADIO_PORT="${QWEN3_GRADIO_PORT:-7860}"
ASR_HEALTH_HOST="${ASR_HEALTH_HOST:-127.0.0.1}"
ALIGNER_HEALTH_HOST="${ALIGNER_HEALTH_HOST:-127.0.0.1}"
GRADIO_HEALTH_HOST="${GRADIO_HEALTH_HOST:-127.0.0.1}"

COMMON_CUDA_VISIBLE_DEVICES="${QWEN3_CUDA_VISIBLE_DEVICES:-0}"
ASR_CUDA_VISIBLE_DEVICES="${QWEN3_ASR_CUDA_VISIBLE_DEVICES:-$COMMON_CUDA_VISIBLE_DEVICES}"
ALIGNER_CUDA_VISIBLE_DEVICES="${QWEN3_ALIGNER_CUDA_VISIBLE_DEVICES:-$COMMON_CUDA_VISIBLE_DEVICES}"

cat > "$RUNTIME_DIR/gpu_visible_devices.txt" <<GPUINFO
QWEN3_CUDA_VISIBLE_DEVICES=$COMMON_CUDA_VISIBLE_DEVICES
QWEN3_ASR_CUDA_VISIBLE_DEVICES=$ASR_CUDA_VISIBLE_DEVICES
QWEN3_ALIGNER_CUDA_VISIBLE_DEVICES=$ALIGNER_CUDA_VISIBLE_DEVICES
GPUINFO

python -m pip install -U pip "setuptools<81.0.0,>=77.0.3" wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install modelscope
if [[ "${QWEN3_GRADIO_ENABLE:-1}" == "1" ]]; then
  python -m pip install -r requirements-client.txt
fi

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
  CUDA_VISIBLE_DEVICES="$ALIGNER_CUDA_VISIBLE_DEVICES" setsid python -m deploy.forced_aligner_server \
    --host "$ALIGNER_HOST" \
    --port "$ALIGNER_PORT" \
    --model "$ALIGNER_MODEL_DIR" \
    --served-model-name Qwen3-ForcedAligner-0.6B \
    --runner pooling \
    --pooler-config '{"task":"token_classify"}' \
    --hf-overrides '{"architectures":["Qwen3ASRForcedAlignerForTokenClassification"]}' \
    --gpu-memory-utilization "${QWEN3_ALIGNER_GPU_MEMORY_UTILIZATION:-0.10}" \
    --kv-cache-memory-bytes "${QWEN3_ALIGNER_KV_CACHE_MEMORY_BYTES:-2G}" \
    --cpu-offload-gb "${QWEN3_ALIGNER_CPU_OFFLOAD_GB:-0}" \
    --disable-log-requests \
    > "$LOG_DIR/aligner_server.log" 2>&1 &
  echo $! > "$RUNTIME_DIR/aligner_server.pid"
fi

ln -sfn "$LOG_DIR/aligner_server.log" "$RUNTIME_DIR/aligner_server.log"
wait_for_http "http://$ALIGNER_HEALTH_HOST:$ALIGNER_PORT/health" "$RUNTIME_DIR/health_aligner.json" "$LOG_DIR/aligner_server.log"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits > "$RUNTIME_DIR/gpu_after_aligner.txt" || true

if [[ -f "$RUNTIME_DIR/asr_server.pid" ]] && kill -0 "$(cat "$RUNTIME_DIR/asr_server.pid")" 2>/dev/null; then
  echo "ASR server already running: $(cat "$RUNTIME_DIR/asr_server.pid")"
else
  ASR_EXTRA_ARGS=()
  if [[ "${QWEN3_ASR_ENFORCE_EAGER:-0}" == "1" ]]; then
    ASR_EXTRA_ARGS+=(--enforce-eager)
  fi
  CUDA_VISIBLE_DEVICES="$ASR_CUDA_VISIBLE_DEVICES" setsid qwen3-asr-native-server \
    --host "$ASR_HOST" \
    --port "$ASR_PORT" \
    --model-path "$ASR_MODEL_DIR" \
    --gpu-memory-utilization "${QWEN3_ASR_GPU_MEMORY_UTILIZATION:-0.30}" \
    --kv-cache-memory-bytes "${QWEN3_ASR_KV_CACHE_MEMORY_BYTES:-8G}" \
    --cpu-offload-gb "${QWEN3_ASR_CPU_OFFLOAD_GB:-0}" \
    --max-model-len "${QWEN3_ASR_MAX_MODEL_LEN:-65536}" \
    --dtype "${QWEN3_ASR_DTYPE:-auto}" \
    --max-new-tokens "${QWEN3_ASR_MAX_NEW_TOKENS:-128}" \
    --chunk-size-sec "${QWEN3_ASR_CHUNK_SIZE_SEC:-1.0}" \
    --unfixed-chunk-num "${QWEN3_ASR_UNFIXED_CHUNK_NUM:-2}" \
    --unfixed-token-num "${QWEN3_ASR_UNFIXED_TOKEN_NUM:-5}" \
    --audio-queue-size "${QWEN3_ASR_AUDIO_QUEUE_SIZE:-8}" \
    --send-queue-size "${QWEN3_ASR_SEND_QUEUE_SIZE:-32}" \
    --decode-timeout-sec "${QWEN3_ASR_DECODE_TIMEOUT_SEC:-0}" \
    --enable-offline-api \
    --offline-num-threads "${QWEN3_ASR_OFFLINE_NUM_THREADS:-2}" \
    --vad-target-segment-s "${QWEN3_ASR_VAD_TARGET_SEGMENT_S:-45}" \
    --vad-max-segment-s "${QWEN3_ASR_VAD_MAX_SEGMENT_S:-60}" \
    --max-concurrent-asr-jobs "${QWEN3_ASR_MAX_CONCURRENT_JOBS:-1}" \
    --aligner-mode "${QWEN3_ALIGNER_MODE:-remote}" \
    --aligner-base-url "${QWEN3_ALIGNER_BASE_URL:-http://127.0.0.1:$ALIGNER_PORT}" \
    --aligner-api-key "${QWEN3_ALIGNER_API_KEY:-EMPTY}" \
    --aligner-timeout-s "${QWEN3_ALIGNER_TIMEOUT_S:-120}" \
    --aligner-timestamp-segment-time-ms "${QWEN3_ALIGNER_TIMESTAMP_SEGMENT_TIME_MS:-80}" \
    --aligner-model-path Qwen3-ForcedAligner-0.6B \
    "${ASR_EXTRA_ARGS[@]}" \
    > "$LOG_DIR/asr_server.log" 2>&1 &
  echo $! > "$RUNTIME_DIR/asr_server.pid"
fi

ln -sfn "$LOG_DIR/asr_server.log" "$RUNTIME_DIR/asr_server.log"
wait_for_http "http://$ASR_HEALTH_HOST:$ASR_PORT/health" "$RUNTIME_DIR/health_asr.json" "$LOG_DIR/asr_server.log" || {
  bash scripts/stop_native_asr.sh --env-file "$ENV_FILE" || true
  exit 1
}
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits > "$RUNTIME_DIR/gpu_after_asr.txt" || true

if [[ "${QWEN3_GRADIO_ENABLE:-1}" == "1" ]]; then
  if [[ -f "$RUNTIME_DIR/gradio_server.pid" ]] && kill -0 "$(cat "$RUNTIME_DIR/gradio_server.pid")" 2>/dev/null; then
    echo "Gradio already running: $(cat "$RUNTIME_DIR/gradio_server.pid")"
  else
    ensure_tcp_port_available "$GRADIO_PORT" "Gradio"
    GRADIO_SHARE_ARGS=()
    if [[ "${QWEN3_GRADIO_SHARE:-0}" == "1" ]]; then
      GRADIO_SHARE_ARGS+=(--share)
    fi
    GRADIO_SSL_ARGS=()
    GRADIO_HEALTH_SCHEME=http
    if [[ -n "${QWEN3_GRADIO_SSL_CERTFILE:-}" || -n "${QWEN3_GRADIO_SSL_KEYFILE:-}" ]]; then
      if [[ -z "${QWEN3_GRADIO_SSL_CERTFILE:-}" || -z "${QWEN3_GRADIO_SSL_KEYFILE:-}" ]]; then
        echo "QWEN3_GRADIO_SSL_CERTFILE and QWEN3_GRADIO_SSL_KEYFILE must be configured together." >&2
        exit 1
      fi
      GRADIO_SSL_ARGS+=(--ssl-certfile "$QWEN3_GRADIO_SSL_CERTFILE" --ssl-keyfile "$QWEN3_GRADIO_SSL_KEYFILE")
      GRADIO_HEALTH_SCHEME=https
    fi
    if [[ -n "${QWEN3_GRADIO_SSL_KEYFILE_PASSWORD:-}" ]]; then
      GRADIO_SSL_ARGS+=(--ssl-keyfile-password "$QWEN3_GRADIO_SSL_KEYFILE_PASSWORD")
    fi
    if [[ "${QWEN3_GRADIO_SSL_NO_VERIFY:-0}" == "1" ]]; then
      GRADIO_SSL_ARGS+=(--ssl-no-verify)
    fi
    setsid qwen3-asr-gradio \
      --server "${QWEN3_GRADIO_SERVER:-http://127.0.0.1:$ASR_PORT}" \
      --host "$GRADIO_HOST" \
      --port "$GRADIO_PORT" \
      --chunk-size-sec "${QWEN3_ASR_CHUNK_SIZE_SEC:-1.0}" \
      --unfixed-chunk-num "${QWEN3_ASR_UNFIXED_CHUNK_NUM:-2}" \
      --unfixed-token-num "${QWEN3_ASR_UNFIXED_TOKEN_NUM:-5}" \
      --realtime-language-1 "${QWEN3_GRADIO_REALTIME_LANGUAGE_1:-Chinese}" \
      --realtime-language-2 "${QWEN3_GRADIO_REALTIME_LANGUAGE_2:-English}" \
      "${GRADIO_SHARE_ARGS[@]}" \
      "${GRADIO_SSL_ARGS[@]}" \
      > "$LOG_DIR/gradio_server.log" 2>&1 &
    echo $! > "$RUNTIME_DIR/gradio_server.pid"
  fi
  ln -sfn "$LOG_DIR/gradio_server.log" "$RUNTIME_DIR/gradio_server.log"
  wait_for_http "$GRADIO_HEALTH_SCHEME://$GRADIO_HEALTH_HOST:$GRADIO_PORT" "$RUNTIME_DIR/health_gradio.html" "$LOG_DIR/gradio_server.log" 180 "${QWEN3_GRADIO_SSL_NO_VERIFY:-0}"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits > "$RUNTIME_DIR/gpu_after_gradio.txt" || true
fi

cat <<EOFOUT
部署完成：

GPU:
  QWEN3_CUDA_VISIBLE_DEVICES=$COMMON_CUDA_VISIBLE_DEVICES
  ASR CUDA_VISIBLE_DEVICES=$ASR_CUDA_VISIBLE_DEVICES
  ForcedAligner CUDA_VISIBLE_DEVICES=$ALIGNER_CUDA_VISIBLE_DEVICES

ASR:
  local:  http://127.0.0.1:$ASR_PORT
  remote: http://<server-ip>:$ASR_PORT
  ws:     ws://<server-ip>:$ASR_PORT/ws/stream

ForcedAligner:
  local:  http://127.0.0.1:$ALIGNER_PORT
  remote: http://<server-ip>:$ALIGNER_PORT

Gradio:
  local:  http://127.0.0.1:$GRADIO_PORT
  remote: http://<server-ip>:$GRADIO_PORT

Runtime:
  $RUNTIME_DIR

Logs:
  $LOG_DIR/asr_server.log
  $LOG_DIR/gradio_server.log
  $LOG_DIR/aligner_server.log
EOFOUT
