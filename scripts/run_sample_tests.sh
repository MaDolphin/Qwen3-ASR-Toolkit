#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_URL="${API_URL:-http://127.0.0.1:18000/api/v1/offline/transcribe}"
WS_URL="${WS_URL:-ws://127.0.0.1:18000/ws/v1/realtime/transcribe}"

cd "${ROOT_DIR}"

echo "== Health Check =="
curl -s http://127.0.0.1:18000/health || true
echo

echo "== Offline Short Audio =="
python -m qwen3_asr_toolkit.offline_cli \
  -i sample/sample_0.mp3 \
  -u "${API_URL}" \
  --save-text

echo
echo "== Offline Short Audio With Aligner =="
python -m qwen3_asr_toolkit.offline_cli \
  -i sample/sample_0.mp3 \
  -u "${API_URL}" \
  --use-forced-aligner

echo
echo "== Offline Long Audio =="
python -m qwen3_asr_toolkit.offline_cli \
  -i sample/deutsch.mp3 \
  -u "${API_URL}"

echo
echo "== Realtime WebSocket =="
ALL_PROXY= \
all_proxy= \
HTTP_PROXY= \
http_proxy= \
HTTPS_PROXY= \
https_proxy= \
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
python -m qwen3_asr_toolkit.realtime_cli \
  -i sample/sample_0.mp3 \
  -u "${WS_URL}" \
  --chunk-ms 500
