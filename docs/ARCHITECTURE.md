# 架构说明

## 总体架构

```text
CLI / HTTP / WebSocket
        │
        ▼
┌──────────────────────────────────────┐
│ Qwen3-ASR Native Server               │
│ - Qwen3-ASR-1.7B                      │
│ - HTTP offline                        │
│ - WebSocket streaming                 │
└───────────────────┬──────────────────┘
                    │ remote aligner
                    ▼
┌──────────────────────────────────────┐
│ Qwen3-ForcedAligner vLLM Server       │
│ - Qwen3-ForcedAligner-0.6B            │
│ - /tokenize + /pooling                │
└──────────────────────────────────────┘
```

ASR Native Server 通过 `Qwen3ASRModel.LLM()` 加载 `Qwen3-ASR-1.7B`，同时服务离线 HTTP 和在线 WebSocket。

## 离线 HTTP 数据流

```text
HTTP upload
  -> 临时文件
  -> load_audio()
  -> VAD / 固定最大段长切分
  -> NativeQwenASRAdapter
  -> 共享 Qwen3ASRModel.transcribe()
  -> 合并 segments
  -> optional remote forced aligner
  -> JSON response
```

## 在线 WebSocket 数据流

```text
receiver_task:
  receive start / binary / finish
  binary audio -> audio_queue
  ack -> send_queue

processor_task:
  audio_queue -> streaming_transcribe()
  partial -> send_queue
  finish -> finish_streaming_transcribe()
  final -> send_queue

sender_task:
  send_queue -> websocket.send_json()
```

## 长音频在线策略

单个 WebSocket session 推荐不超过 `120s`。更长音频使用：

```text
120s window + 0s overlap + sequential WebSocket sessions
```

## 显存策略

默认部署面向已被其他模型占用部分显存的 L20：

```text
ASR gpu_memory_utilization=0.30
ASR kv_cache_memory_bytes=8G
ForcedAligner gpu_memory_utilization=0.10
ForcedAligner kv_cache_memory_bytes=2G
```

## 当前限制

- 在线方案是服务层 streaming/windowed 策略，不声称 KV-cache 级 true incremental streaming。
- 无 token-level streaming。
- 默认长音频在线使用 windowed session。


## 客户端分层

```text
client/cli      -> 命令行客户端，调用 HTTP / WebSocket API
client/gradio   -> Web UI 客户端，支持离线上传和浏览器麦克风实时转写
examples/       -> Python 调用示例和验证脚本
deploy/         -> 服务端部署入口、ForcedAligner wrapper、systemd
```

`client/` 和 `examples/` 都不会加载 `Qwen3-ASR-1.7B`，只访问已经部署好的 Native ASR Server。


## 并发模型

FastAPI 可以同时接受多个 HTTP 和 WebSocket 连接，但所有 ASR 模型推理会进入同一个优先级队列，由单个 ASR worker 串行执行。这样可以避免同一个 vLLM `LLM` 实例并行 `generate` 时触发 EngineCore 异常。

```text
HTTP offline segment
  -> NativeQwenASRAdapter
  -> ASRScheduler(priority=10, label=offline-transcribe)

WebSocket chunk/final
  -> websocket processor_task
  -> ASRScheduler(priority=0, label=ws-*)

ASRScheduler
  -> single worker thread
  -> shared Qwen3ASRModel.LLM
```

关键字段：

- `max_concurrent_asr_jobs`：实际模型执行并发，固定为 `1`。
- `requested_max_concurrent_asr_jobs`：启动参数中请求的值，仅用于观测和兼容。
- `asr_scheduler=priority-single-worker`：表示启用单 worker 优先级调度。
- `realtime_priority=true`：表示 WebSocket 实时任务优先于离线分段任务。
- `asr_scheduler_queue_size`：当前等待进入模型的任务数。

`QWEN3_ASR_OFFLINE_NUM_THREADS` 只影响离线长音频分段准备和提交任务的并行度，不代表模型推理并行。若实时延迟明显增大或 partial 堆积，优先将它降到 `1`。Gradio 部署在同一台服务器上，但只作为客户端，不使用 GPU。
