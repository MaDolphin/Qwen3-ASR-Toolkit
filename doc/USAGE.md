# 使用文档

本文档说明如何使用 `Qwen3-ASR-Toolkit` 提供的离线转写、实时转写、本地 CLI 和云端接口。

## 1. 使用方式总览

本项目支持四类使用方式：

1. 离线 REST API
2. 实时 WebSocket API
3. 离线 CLI
4. 实时 CLI

另外，还保留了“本地直连 vLLM”的离线 CLI 调用方式。

## 2. 离线 REST API

### 2.1 接口地址

```text
POST /api/v1/offline/transcribe
```

### 2.2 请求格式

- `Content-Type: multipart/form-data`

表单字段：

- `audio_file`
  - 必填
  - 音频或视频文件
- `context`
  - 可选
  - 给 ASR 的上下文提示词
- `use_forced_aligner`
  - 可选
  - `true` 或 `false`
  - 只有离线接口支持该参数

### 2.3 curl 示例

```bash
curl -X POST "http://127.0.0.1:18000/api/v1/offline/transcribe" \
  -F "audio_file=@sample/sample_0.mp3" \
  -F "context=会议纪要" \
  -F "use_forced_aligner=true"
```

### 2.4 返回结果说明

典型返回字段：

- `language`
- `text`
- `segment_count`
- `audio_duration_sec`
- `segments`
- `forced_aligner`

其中：

- `segments`
  - 表示长音频切分后的分段结果
- `forced_aligner.requested`
  - 是否请求了对齐
- `forced_aligner.available`
  - 对齐是否执行成功
- `forced_aligner.items`
  - 词级或字级时间戳条目

## 3. 实时 WebSocket API

### 3.1 接口地址

```text
ws://127.0.0.1:18000/ws/v1/realtime/transcribe
```

### 3.2 重要说明

- 实时模式不支持强制对齐
- 实时模式只返回准实时增量文本、段落完成事件和最终文本
- 输入音频需要是：
  - `float32`
  - `16k`
  - 单声道 PCM 原始字节流

### 3.3 消息流程

#### 服务端连接成功后

服务端先返回：

```json
{
  "event": "ready",
  "protocol_version": "2.0",
  "session_id": "...",
  "sample_rate": 16000,
  "audio_format": "float32le_pcm_mono_16k",
  "realtime_alignment_supported": false,
  "session_defaults": {
    "decode_interval_ms": 600,
    "min_chunk_ms": 200,
    "finalize_silence_ms": 600,
    "max_segment_sec": 20.0
  }
}
```

#### 客户端启动会话

发送：

```json
{
  "event": "start",
  "context": "会议场景",
  "decode_interval_ms": 600,
  "min_chunk_ms": 200,
  "finalize_silence_ms": 600,
  "max_segment_sec": 20.0
}
```

#### 服务端返回 started

```json
{
  "event": "started",
  "session_id": "...",
  "session": {
    "context": "会议场景",
    "decode_interval_ms": 600,
    "min_chunk_ms": 200,
    "finalize_silence_ms": 600,
    "max_segment_sec": 20.0
  }
}
```

#### 客户端持续发送音频块

- 二进制帧
- 内容为 `float32` PCM 字节流

#### 服务端持续返回 partial

```json
{
  "event": "partial",
  "updated": true,
  "language": "Chinese",
  "text": "当前增量结果",
  "committed_text": "已确认部分",
  "live_text": "当前未完成语音段"
}
```

#### 服务端在检测到静音提交后返回 `segment_final`

```json
{
  "event": "segment_final",
  "language": "Chinese",
  "text": "已确认完整文本",
  "committed_text": "已确认完整文本",
  "live_text": "",
  "segment_text": "本次完成的语音段",
  "segment_index": 0
}
```

#### 客户端发送结束事件

```json
{
  "event": "finish"
}
```

#### 服务端返回最终结果

```json
{
  "event": "final",
  "language": "Chinese",
  "text": "最终结果",
  "committed_text": "最终结果",
  "live_text": ""
}
```

### 3.4 边界说明

- WebSocket 实时转写不支持 `use_forced_aligner`
- 若在 `start` 消息中传入对齐相关字段，服务端会直接返回错误
- 当前实时实现是“队列解耦 + 当前语音段增量识别 + 静音提交”的准实时模式
- 它不依赖远端 vLLM 的官方 `/v1/realtime` 端点

## 4. 离线 CLI

### 4.1 作用

离线 CLI 会调用 REST API，而不是直接本地推理。

入口：

```bash
python -m qwen3_asr_toolkit.offline_cli --help
```

### 4.2 示例

基础转写：

```bash
python -m qwen3_asr_toolkit.offline_cli \
  -i sample/sample_0.mp3 \
  -u http://127.0.0.1:18000/api/v1/offline/transcribe
```

带对齐：

```bash
python -m qwen3_asr_toolkit.offline_cli \
  -i sample/sample_0.mp3 \
  -u http://127.0.0.1:18000/api/v1/offline/transcribe \
  --use-forced-aligner
```

保存文本：

```bash
python -m qwen3_asr_toolkit.offline_cli \
  -i sample/sample_0.mp3 \
  -u http://127.0.0.1:18000/api/v1/offline/transcribe \
  --save-text
```

## 5. 实时 CLI

### 5.1 作用

实时 CLI 会读取本地音频文件，把它切成小块后通过 WebSocket 推送给服务端，模拟实时流式输入。

入口：

```bash
python -m qwen3_asr_toolkit.realtime_cli --help
```

### 5.2 示例

```bash
python -m qwen3_asr_toolkit.realtime_cli \
  -i sample/sample_0.mp3 \
  -u ws://127.0.0.1:18000/ws/v1/realtime/transcribe \
  --chunk-ms 300 \
  --decode-interval-ms 600 \
  --finalize-silence-ms 600
```

模拟真实时钟：

```bash
python -m qwen3_asr_toolkit.realtime_cli \
  -i sample/sample_0.mp3 \
  -u ws://127.0.0.1:18000/ws/v1/realtime/transcribe \
  --chunk-ms 500 \
  --decode-interval-ms 600 \
  --finalize-silence-ms 600 \
  --simulate-realtime
```

## 6. 本地直连 vLLM CLI

### 6.1 作用

这个模式不经过 REST/WS 服务，而是直接从本地 CLI 调用 vLLM OpenAI 兼容接口。

入口：

```bash
python -m qwen3_asr_toolkit.call_api --help
```

### 6.2 示例

```bash
python -m qwen3_asr_toolkit.call_api \
  -i sample/sample_0.mp3 \
  -url http://172.28.245.150:10010/v1 \
  -m Qwen3-ASR-1.7B
```

带 SRT 输出：

```bash
python -m qwen3_asr_toolkit.call_api \
  -i sample/deutsch.mp3 \
  -url http://172.28.245.150:10010/v1 \
  -m Qwen3-ASR-1.7B \
  -srt
```

带离线对齐：

```bash
python -m qwen3_asr_toolkit.call_api \
  -i sample/sample_0.mp3 \
  -url http://172.28.245.150:10010/v1 \
  -m Qwen3-ASR-1.7B \
  --use-forced-aligner
```

## 7. 行为边界说明

### 7.1 长音频

长音频会自动走 VAD。

系统保证：

- 每个最终切分段 `<= 60s`
- 即使 VAD 失败，也会回退到固定长度切分

### 7.2 对齐功能

对齐只在离线模式生效：

- 离线 REST
- 离线 CLI
- 本地直连 CLI

实时模式不做对齐：

- WebSocket API 不支持对齐参数
- 实时 CLI 不支持对齐参数

### 7.3 对齐返回质量

当前对齐已经可以返回真实时间戳，但因为远端服务是 `token_classify` 输出，部分尾部 token 可能出现相同结束时间。这属于当前服务特性，需要后续继续优化后处理逻辑。

## 8. 推荐使用顺序

如果你是第一次接这个项目，推荐顺序：

1. 先跑 `curl /health`
2. 再跑离线短音频
3. 再跑离线长音频
4. 最后跑实时 WebSocket

这样排查问题最省时间。
