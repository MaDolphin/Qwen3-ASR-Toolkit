#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
RUNTIME_DIR="${ROOT_DIR}/runtime"
PID_FILE="${RUNTIME_DIR}/server.pid"
LOG_FILE="${RUNTIME_DIR}/server.log"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18000}"

mkdir -p "${RUNTIME_DIR}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing file: ${path}" >&2
    exit 1
  fi
}

ensure_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi
}

install_deps() {
  ensure_venv
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install -U pip
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
  python -m pip install -e "${ROOT_DIR}"
}

is_running() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if kill -0 "${pid}" >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

start_server() {
  require_file "${ROOT_DIR}/.env"
  ensure_venv
  if is_running; then
    echo "Server is already running. PID=$(cat "${PID_FILE}")"
    return 0
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  cd "${ROOT_DIR}"
  nohup python -m qwen3_asr_toolkit.server --host "${HOST}" --port "${PORT}" >"${LOG_FILE}" 2>&1 &
  echo $! >"${PID_FILE}"
  sleep 2

  if is_running; then
    echo "Server started. PID=$(cat "${PID_FILE}")"
    echo "Log file: ${LOG_FILE}"
  else
    echo "Server failed to start. Check logs: ${LOG_FILE}" >&2
    exit 1
  fi
}

stop_server() {
  if ! is_running; then
    echo "Server is not running."
    rm -f "${PID_FILE}"
    return 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  kill "${pid}" >/dev/null 2>&1 || true
  sleep 1

  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill -9 "${pid}" >/dev/null 2>&1 || true
  fi

  rm -f "${PID_FILE}"
  echo "Server stopped."
}

status_server() {
  if is_running; then
    echo "Server is running. PID=$(cat "${PID_FILE}")"
  else
    echo "Server is not running."
  fi
}

logs_server() {
  require_file "${LOG_FILE}"
  tail -n 100 -f "${LOG_FILE}"
}

healthcheck() {
  curl -s "http://127.0.0.1:${PORT}/health" || true
}

usage() {
  cat <<EOF
Usage: bash scripts/deploy_server.sh <command>

Commands:
  install     Create .venv and install project dependencies
  start       Start the ASR service in background
  stop        Stop the running ASR service
  restart     Restart the ASR service
  status      Show service status
  logs        Tail service logs
  health      Call local /health endpoint
EOF
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    install)
      install_deps
      ;;
    start)
      start_server
      ;;
    stop)
      stop_server
      ;;
    restart)
      stop_server
      start_server
      ;;
    status)
      status_server
      ;;
    logs)
      logs_server
      ;;
    health)
      healthcheck
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
