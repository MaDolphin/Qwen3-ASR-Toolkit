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
ASR gpu_memory_utilization=0.50
ASR kv_cache_memory_bytes=8G
ForcedAligner gpu_memory_utilization=0.10
ForcedAligner kv_cache_memory_bytes=2G
```

## 当前限制

- 在线方案是服务层 streaming/windowed 策略，不声称 KV-cache 级 true incremental streaming。
- 无 token-level streaming。
- 默认长音频在线使用 windowed session。
