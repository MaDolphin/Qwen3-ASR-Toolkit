#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash scripts/stop_native_asr.sh"
  exit 0
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_DIR/runtime/native_deploy}"

stop_pid_file() {
  local file="$1"
  local name="$2"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name pid=$pid"
      kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      kill -9 -- "-$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
  fi
}

stop_pid_file "$RUNTIME_DIR/asr_server.pid" "ASR server"
stop_pid_file "$RUNTIME_DIR/aligner_server.pid" "ForcedAligner server"

nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits || true
