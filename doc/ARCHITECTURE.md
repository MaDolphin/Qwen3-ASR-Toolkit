# 架构说明

## 总体架构

```text
HTTP offline ─┐
              ├── Unified Native ASR Server ── 单个 Qwen3-ASR-1.7B 实例
WS streaming ─┘
```

服务入口：

```text
deploy/vllm_streaming_server_native.py
```

该进程通过 `Qwen3ASRModel.LLM()` 加载一次 `Qwen3-ASR-1.7B`，并复用同一个模型对象处理离线 HTTP 和在线 WebSocket 请求。

## 离线 HTTP 数据流

```text
HTTP upload
  -> 临时文件
  -> load_audio()
  -> VAD / 固定最大段长切分
  -> NativeQwenASRAdapter
  -> 共享 Qwen3ASRModel.transcribe()
  -> 合并 segments
  -> optional forced aligner metadata
  -> JSON response
```

关键约束：

- 默认 `vad_target_segment_s=45`。
- 默认 `vad_max_segment_s=60`。
- `offline_num_threads=1`。
- 所有模型调用受 `max_concurrent_asr_jobs` 控制。

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

这种结构避免 `model.generate()` 阻塞 WebSocket receive，并通过队列提供应用层背压。

## 长音频在线策略

单个 WebSocket session 推荐不超过 `120s`。更长音频使用：

```text
120s window + 0s overlap + sequential WebSocket sessions
```

验证脚本：

```text
deploy/test_native_streaming_windowed_harness.py
```

## Forced Aligner

`Qwen3-ForcedAligner-0.6B` 是独立模型，不属于“ASR 模型只加载一次”的约束。

默认：

```text
aligner_mode=disabled
```

生产建议优先使用远端 aligner 服务：

```text
aligner_mode=remote
```

## 当前限制

- 当前在线方案不是 KV-cache 级 true incremental streaming。
- 当前底层仍会基于窗口或累计音频进行转写。
- 无 token-level streaming。
- 无 `segment_final`。
- 默认无 forced aligner 时间戳。
