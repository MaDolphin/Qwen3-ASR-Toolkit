# 验证说明

所有验证默认使用：

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen-asr
```

## 1. 静态检查

```bash
python -m py_compile \
  deploy/vllm_streaming_server_native.py \
  deploy/streaming_utils.py \
  deploy/test_native_streaming_ws_harness.py \
  deploy/test_native_streaming_windowed_harness.py \
  qwen3_asr_toolkit/audio_tools.py \
  qwen3_asr_toolkit/offline_transcriber.py \
  qwen3_asr_toolkit/forced_aligner_client.py
```

## 2. 单元测试

```bash
pytest -q
```

## 3. 服务启动验证

```bash
qwen3-asr-native-server \
  --host 0.0.0.0 \
  --port 10012 \
  --model-path /workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.50 \
  --enable-offline-api \
  --offline-num-threads 1 \
  --aligner-mode remote
```

## 4. Health 验证

```bash
curl -s http://127.0.0.1:10012/health
```

通过标准：

```text
status == ok
capabilities.offline_http == true
capabilities.native_websocket == true
capabilities.asr_model_loaded_once == true
```

## 5. 离线短音频验证

```bash
curl -X POST "http://127.0.0.1:10012/api/v1/offline/transcribe" \
  -F "audio_file=@sample/sample_0.mp3" \
  -F "context=" \
  -F "use_forced_aligner=false"
```

通过标准：

```text
text 非空
language 非空
segment_count >= 1
asr_model_loaded_once == true
```

## 6. 离线长音频验证

```bash
curl -X POST "http://127.0.0.1:10012/api/v1/offline/transcribe" \
  -F "audio_file=@sample/sample_2.m4a" \
  -F "context=" \
  -F "use_forced_aligner=false" \
  -o runtime/native_validation/offline_sample_2.json
```

通过标准：

```text
text 非空
segment_count > 1
最大 segment duration <= 60s
```

## 7. WebSocket 120s 验证

```bash
python deploy/test_native_streaming_ws_harness.py \
  --uri ws://127.0.0.1:10012/ws/stream \
  --input sample/sample_2.m4a \
  --reference sample/sample_2.txt \
  --output runtime/native_validation/sample_2_120s.json \
  --case-label sample_2_120s \
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
```

通过标准：

```text
passed == true
chunks_sent == ack_count
partial_count > 0
final.text 非空
```

## 8. WebSocket windowed 全量验证

```bash
python deploy/test_native_streaming_windowed_harness.py \
  --uri ws://127.0.0.1:10012/ws/stream \
  --input sample/sample_2.m4a \
  --reference sample/sample_2.txt \
  --output runtime/native_validation/sample_2_windowed.json \
  --report runtime/native_validation/SAMPLE_2_WINDOWED_REPORT.md \
  --case-dir runtime/native_validation/sample_2_windowed_cases \
  --case-label sample_2_windowed \
  --window-sec 120 \
  --overlap-sec 0 \
  --chunk-ms 500
```

通过标准：

```text
passed == true
所有窗口通过
aggregate_text_len > 0
无 server error
```

## 9. 单模型加载验证

```bash
nvidia-smi
ps -ef | grep -E "qwen3-asr-native-server|vllm_streaming_server_native|VLLM::EngineCore|vllm serve"
```

通过标准：

- 没有独立 `vllm serve models/Qwen3-ASR-1.7B`。
- 没有第二个 Native server。
- `Qwen3-ASR-1.7B` 只被当前服务加载一次。

## CLI 验证

```bash
qwen3-asr-offline-cli \
  --input-file sample/sample_0.mp3 \
  --output-json runtime/validation/offline_cli.json

qwen3-asr-stream-cli \
  --input-file sample/sample_2.m4a \
  --duration-sec 120 \
  --output-json runtime/validation/ws_cli_120s.json \
  --realtime
```

## 一键功能测试

```bash
bash scripts/test_native_asr_functional.sh
```
