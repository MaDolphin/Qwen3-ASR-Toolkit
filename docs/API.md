# API 文档

统一 Native 服务默认监听：

```text
http://127.0.0.1:10012
ws://127.0.0.1:10012/ws/stream
```

## GET /health

用于检查模型和服务状态。

```bash
curl -s http://127.0.0.1:10012/health
```

示例响应：

```json
{
  "status": "ok",
  "model": "models/Qwen3-ASR-1.7B",
  "backend": "vllm",
  "message": "Model is ready.",
  "capabilities": {
    "offline_http": true,
    "native_websocket": true,
    "forced_aligner": "remote",
    "max_concurrent_asr_jobs": 1,
    "requested_max_concurrent_asr_jobs": 1,
    "asr_scheduler": "priority-single-worker",
    "realtime_priority": true,
    "asr_scheduler_queue_size": 0,
    "asr_model_loaded_once": true,
    "offline_num_threads": 1,
    "vad_target_segment_s": 45,
    "vad_max_segment_s": 60
  }
}
```

## POST /api/v1/offline/transcribe

离线文件转写接口。

### 请求

```bash
curl -X POST "http://127.0.0.1:10012/api/v1/offline/transcribe" \
  -F "audio_file=@sample/sample_0.mp3" \
  -F "context=" \
  -F "use_forced_aligner=false"
```

字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `audio_file` | file | 是 | 音频文件，支持常见格式。 |
| `context` | string | 否 | 提示上下文。 |
| `use_forced_aligner` | bool | 否 | 是否请求 forced aligner。默认 `false`。 |

### 响应

```json
{
  "language": "Chinese",
  "text": "...",
  "segment_count": 1,
  "audio_duration_sec": 9.0,
  "segments": [
    {
      "index": 0,
      "start_sec": 0.0,
      "end_sec": 9.0,
      "duration_sec": 9.0,
      "language": "Chinese",
      "text": "..."
    }
  ],
  "forced_aligner": {
    "requested": false,
    "available": false,
    "message": "Forced aligner not requested.",
    "items": []
  },
  "file_name": "sample_0.mp3",
  "backend": "native-vllm",
  "asr_model_loaded_once": true
}
```

### 错误

| HTTP 状态码 | 场景 |
| --- | --- |
| `400` | 请求参数不合法，或 `local-lazy` forced aligner 尚未启用。 |
| `404` | 离线 HTTP API 被关闭。 |
| `500` | 音频加载、分段或转写失败。 |
| `503` | ASR 模型尚未加载。 |

## WS /ws/stream

在线 WebSocket 转写接口。

### start

客户端连接后发送：

```json
{
  "event": "start",
  "stream": true,
  "context": "",
  "chunk_size_sec": 1.0,
  "unfixed_chunk_num": 2,
  "unfixed_token_num": 5
}
```

服务端返回：

```json
{
  "event": "started",
  "stream": true,
  "sample_rate": 16000,
  "audio_format": "float32le_pcm_mono_16k",
  "chunk_size_sec": 1.0,
  "unfixed_chunk_num": 2,
  "unfixed_token_num": 5
}
```

### binary audio

客户端发送二进制音频：

```text
float32le PCM, mono, 16000 Hz
```

服务端确认接收：

```json
{
  "event": "ack",
  "received_samples": 8000,
  "total_samples": 8000,
  "duration_sec": 0.5
}
```

服务端返回中间结果：

```json
{
  "event": "partial",
  "language": "Chinese",
  "text": "...",
  "chunk_id": 1
}
```

### finish

客户端发送：

```json
{
  "event": "finish"
}
```

服务端返回最终结果：

```json
{
  "event": "final",
  "language": "Chinese",
  "text": "...",
  "chunk_id": 120
}
```

### error

```json
{
  "event": "error",
  "message": "..."
}
```

## CLI 与 API 对应关系

| 使用方式 | 底层接口 |
| --- | --- |
| `qwen3-asr-offline-cli --server http://服务器IP:10012` | `POST /api/v1/offline/transcribe` |
| `qwen3-asr-stream-cli --server http://服务器IP:10012` | `WS /ws/stream` |
| `qwen3-asr-cli health --server http://服务器IP:10012` | `GET /health` |


## 客户端对应关系

| 客户端 | 底层接口 |
| --- | --- |
| `qwen3-asr-offline-cli` | `POST /api/v1/offline/transcribe` |
| `qwen3-asr-stream-cli` | `WS /ws/stream` |
| `qwen3-asr-cli health` | `GET /health` |
| Gradio 离线 Tab | `POST /api/v1/offline/transcribe` |
| Gradio 实时 Tab | `WS /ws/stream` |

CLI 和 Gradio 都只是客户端，不加载 ASR 模型。


## 多人访问说明

多人访问不是新的 API，而是服务端调度策略。HTTP 离线和 WebSocket 实时可以同时接入；ASR 模型推理固定串行执行，WebSocket 优先于离线分段任务。Gradio 也是 API 客户端，不新增 ASR 协议。

客户端可以通过 `/health.capabilities.asr_scheduler_queue_size` 观察当前 ASR 模型队列积压；如果该值持续增长，说明请求量超过当前单 worker 推理能力。
