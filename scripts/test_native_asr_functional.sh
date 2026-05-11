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
  echo "Usage: bash scripts/test_native_asr_functional.sh"
  exit 0
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

if [[ -n "${CONDA_ENV:-}" ]]; then
  activate_conda_env
else
  echo "Using current Python environment: $(command -v python)"
fi

ASR_BASE_URL="${ASR_BASE_URL:-http://127.0.0.1:10012}"
ALIGNER_BASE_URL="${ALIGNER_BASE_URL:-http://127.0.0.1:10013}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/runtime/native_deploy_validation}"
mkdir -p "$OUT_DIR"

nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits > "$OUT_DIR/gpu_before_tests.txt" || true
curl -fsS "$ASR_BASE_URL/health" > "$OUT_DIR/health_asr.json"
curl -fsS "$ALIGNER_BASE_URL/health" > "$OUT_DIR/health_aligner.raw"
printf '{"status":"ok","endpoint":"%s/health"}\n' "$ALIGNER_BASE_URL" > "$OUT_DIR/health_aligner.json"

qwen3-asr-offline-cli \
  --input-file sample/sample_0.mp3 \
  --server "$ASR_BASE_URL" \
  --output-json "$OUT_DIR/offline_cli_sample_0.json" \
  --output-text "$OUT_DIR/offline_cli_sample_0.txt"

qwen3-asr-offline-cli \
  --input-file sample/sample_0.mp3 \
  --server "$ASR_BASE_URL" \
  --use-forced-aligner \
  --output-json "$OUT_DIR/offline_cli_sample_0_aligned.json"

qwen3-asr-offline-cli \
  --input-file sample/sample_2.m4a \
  --server "$ASR_BASE_URL" \
  --output-json "$OUT_DIR/offline_cli_sample_2.json" \
  --quiet

python examples/validation/test_native_streaming_ws_harness.py \
  --uri "${ASR_BASE_URL/http:/ws:}/ws/stream" \
  --input sample/sample_2.m4a \
  --reference sample/sample_2.txt \
  --output "$OUT_DIR/websocket_sample_2_120s.json" \
  --event-jsonl "$OUT_DIR/websocket_sample_2_120s_events.jsonl" \
  --case-label deploy_sample_2_120s \
  --start-sec 0 \
  --duration-sec 120 \
  --chunk-ms 500 \
  --chunk-size-sec 1.0 \
  --unfixed-chunk-num 2 \
  --unfixed-token-num 5 \
  --max-inflight-chunks 4 \
  --send-timeout-sec 30 \
  --ack-timeout-sec 120 \
  --receive-timeout-sec 300 \
  --realtime

qwen3-asr-stream-cli \
  --input-file sample/sample_2.m4a \
  --server "$ASR_BASE_URL" \
  --duration-sec 120 \
  --output-json "$OUT_DIR/ws_cli_sample_2_120s.json" \
  --event-jsonl "$OUT_DIR/ws_cli_sample_2_120s_events.jsonl" \
  --realtime \
  --quiet

nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits > "$OUT_DIR/gpu_after_tests.txt" || true

python - <<'PY'
import json
from pathlib import Path
out = Path('runtime/native_deploy_validation')
lines = ['# Native ASR 功能测试报告', '']
for name in ['health_asr.json', 'health_aligner.json', 'offline_cli_sample_0.json', 'offline_cli_sample_0_aligned.json', 'offline_cli_sample_2.json', 'websocket_sample_2_120s.json', 'ws_cli_sample_2_120s.json']:
    path = out / name
    if not path.exists():
        lines.append(f'- `{name}`: 缺失')
        continue
    raw = path.read_text(encoding='utf-8').strip()
    if not raw:
        lines.append(f'- `{name}`: ok')
        continue
    data = json.loads(raw)
    if name.startswith('offline'):
        lines.append(f"- `{name}`: text_len={len(data.get('text',''))}, segments={data.get('segment_count')}, aligner={data.get('forced_aligner',{}).get('available')}")
    elif 'websocket' in name or 'ws_cli' in name:
        lines.append(f"- `{name}`: passed={data.get('passed')}, chunks={data.get('counts',{}).get('chunks_sent')}, ack={data.get('counts',{}).get('ack_count')}")
    else:
        lines.append(f"- `{name}`: ok")
(out / 'FUNCTIONAL_TEST_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY

echo "功能测试完成：$OUT_DIR/FUNCTIONAL_TEST_REPORT.md"
