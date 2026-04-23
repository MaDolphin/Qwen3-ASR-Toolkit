# Qwen3-ASR-Toolkit 云端部署与调用说明（REST + WebSocket）

> 日期：2026-04-16
> 
> 后端推理：vLLM OpenAI 兼容接口

## 1. 启动服务

```bash
cp .env.example .env
# 编辑 .env，至少填写以下字段：
# OPENAI_API_KEY=<your_api_key>
# OPENAI_BASE_URL=http://<your-vllm-host>:<port>/v1
# QWEN3_ASR_MODEL=Qwen3-ASR-1.7B
# QWEN3_ALIGNER_BASE_URL=http://<your-vllm-host>:<aligner_port>
# QWEN3_ALIGNER_API_KEY=<your_api_key>
# QWEN3_ALIGNER_MODEL=Qwen3-ForcedAligner-0.6B

# 可选：VAD 和实时解码参数
# QWEN3_ASR_VAD_TARGET_SEGMENT_S=45
# QWEN3_ASR_VAD_MAX_SEGMENT_S=60
# QWEN3_ASR_STREAM_DECODE_INTERVAL_MS=600
# QWEN3_ASR_STREAM_MIN_CHUNK_MS=200
# QWEN3_ASR_STREAM_FINALIZE_SILENCE_MS=600
# QWEN3_ASR_STREAM_MAX_SEGMENT_S=20
# QWEN3_ALIGNER_TIMEOUT_S=120
# QWEN3_ALIGNER_TIMESTAMP_SEGMENT_TIME_MS=80

qwen3-asr-server --host 0.0.0.0 --port 18000
```

健康检查：

```bash
curl http://127.0.0.1:18000/health
```

## 2. 离线转写（REST）

接口：

- `POST /api/v1/offline/transcribe`
- `multipart/form-data`

表单字段：

- `audio_file`：音频/视频文件（二进制）
- `context`：可选文本上下文
- `use_forced_aligner`：是否请求 `Qwen3-ForcedAligner-0.6B` 对齐（布尔）

示例：

```bash
curl -X POST "http://127.0.0.1:18000/api/v1/offline/transcribe" \
  -F "audio_file=@sample/sample_2.m4a" \
  -F "context=会议纪要" \
  -F "use_forced_aligner=false"
```

说明：

- 长音频会自动走 VAD 切分；
- 每个切分段长度都不会超过 `60s`；
- 当 `use_forced_aligner=true` 且已配置 aligner 服务时，会调用 aligner 的 `/pooling` + `/tokenize` 获取词级时间戳；
- 若 aligner 未配置或调用失败，返回里会给出 `available=false` 与失败原因。

### 离线 CLI（调用 REST）

```bash
qwen3-asr-offline-cli \
  -i sample/sample_2.m4a \
  -u http://127.0.0.1:18000/api/v1/offline/transcribe \
  --context "会议纪要" \
  --save-text
```

启用对齐开关（当前未部署时会返回 unavailable）：

```bash
qwen3-asr-offline-cli \
  -i sample/sample_2.m4a \
  -u http://127.0.0.1:18000/api/v1/offline/transcribe \
  --use-forced-aligner
```

## 3. 实时转写（WebSocket）

接口：

- `ws://<host>:<port>/ws/v1/realtime/transcribe`

消息协议：

1. 服务端连接成功后先发：
   - `{"event":"ready","protocol_version":"2.0","session_id":"...","sample_rate":16000,"audio_format":"float32le_pcm_mono_16k","realtime_alignment_supported":false}`
2. 客户端发 `start`：
   - `{"event":"start","context":"...","decode_interval_ms":600,"finalize_silence_ms":600}`
3. 客户端持续发送二进制音频块：
   - 原始 `float32`、16k、单声道 PCM 字节流
4. 服务端每次返回：
   - `{"event":"partial","updated":true/false,"language":"...","text":"...","committed_text":"...","live_text":"..."}`
5. 当服务端确认一段语音已完成时：
   - `{"event":"segment_final","language":"...","text":"...","segment_text":"..."}`
6. 客户端发 `finish`：
   - `{"event":"finish"}`
7. 服务端返回最终结果：
   - `{"event":"final","language":"...","text":"...","committed_text":"...","live_text":""}`

### 实时 CLI（调用 WS）

```bash
qwen3-asr-stream-cli \
  -i sample/deutsch.mp3 \
  -u ws://127.0.0.1:18000/ws/v1/realtime/transcribe \
  --chunk-ms 300 \
  --decode-interval-ms 600 \
  --finalize-silence-ms 600
```

模拟按真实时钟推流：

```bash
qwen3-asr-stream-cli \
  -i sample/deutsch.mp3 \
  -u ws://127.0.0.1:18000/ws/v1/realtime/transcribe \
  --chunk-ms 500 \
  --simulate-realtime
```

## 4. 本地直连模式（保留）

除了云端 API/WS 模式，保留了本地直连 vLLM 的 CLI：

```bash
qwen3-asr -i sample/sample_0.mp3 -url http://<vllm-host>:<port>/v1 -m Qwen3-ASR-1.7B
```

同样会对长音频启用 VAD，并强制单段 `<=60s`。
