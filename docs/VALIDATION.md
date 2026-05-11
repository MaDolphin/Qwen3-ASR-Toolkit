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
  examples/validation/test_native_streaming_ws_harness.py \
  examples/validation/test_native_streaming_windowed_harness.py \
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
  --gpu-memory-utilization 0.30 \
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
python examples/validation/test_native_streaming_ws_harness.py \
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
python examples/validation/test_native_streaming_windowed_harness.py \
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
qwen3-asr-cli health --server http://服务器IP:10012

qwen3-asr-offline-cli \
  --server http://服务器IP:10012 \
  --input-file sample/sample_0.mp3 \
  --output-json runtime/validation/offline_cli.json

qwen3-asr-stream-cli \
  --server http://服务器IP:10012 \
  --input-file sample/sample_2.m4a \
  --duration-sec 120 \
  --output-json runtime/validation/ws_cli_120s.json \
  --realtime
```

## 一键功能测试

```bash
bash scripts/test_native_asr_functional.sh
```


## Gradio 验证

```bash
qwen3-asr-gradio --server http://127.0.0.1:10012 --host 0.0.0.0 --port 7860
```

验收步骤：

1. 打开 `http://服务器IP:7860`。
2. 离线 Tab 上传 `sample/sample_0.mp3`，确认文本非空。
3. 实时 Tab 授权浏览器麦克风，开始讲话，确认 partial 持续刷新。
4. 点击停止实时转写，确认 final 文本非空。


## .env / GPU / Gradio 验证

```bash
cat .env
cat runtime/native_deploy/gpu_visible_devices.txt
curl -fsS http://127.0.0.1:${QWEN3_ASR_PORT:-10012}/health
curl -fsS http://127.0.0.1:${QWEN3_GRADIO_PORT:-7860} > /tmp/qwen3_gradio.html
```

检查 `/health` 中：

```text
capabilities.max_concurrent_asr_jobs == 1
capabilities.asr_scheduler == priority-single-worker
capabilities.realtime_priority == true
```

## 并发稳定性建议

当前采用单模型单 worker 串行推理，并通过优先级队列让 WebSocket 实时任务优先于离线分段任务。若实时延迟仍明显堆积，请将 `.env` 中 `QWEN3_ASR_OFFLINE_NUM_THREADS` 调整为 `1` 后重测。

建议增加一个并发观察场景：先提交一条离线长音频，再启动 30s WebSocket 测试；预期 WebSocket 连接可以建立并持续收到 partial，离线任务会被延后但不应导致服务崩溃。

## 测试后停止服务

每次真实功能测试完成后执行，或使用 `STOP_AFTER_TEST=1` 自动停止：

```bash
bash scripts/stop_native_asr.sh
```

确保释放 GPU 显存，避免影响后续部署和测试。

```bash
STOP_AFTER_TEST=1 bash scripts/test_native_asr_functional.sh
```
