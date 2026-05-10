# Qwen3-ASR-Toolkit

Qwen3-ASR-Toolkit 是面向生产部署的 Qwen3-ASR Native 服务项目。当前版本只保留一个官方服务入口：**Unified Native ASR Server**。

该服务在一个 GPU 进程中只加载一次 `Qwen3-ASR-1.7B`，同时提供：

- 离线 HTTP 转写：`POST /api/v1/offline/transcribe`
- 在线 WebSocket 转写：`WS /ws/stream`
- 健康检查：`GET /health`

> 不要同时启动 `vllm serve models/Qwen3-ASR-1.7B` 和本项目 Native server。否则 `Qwen3-ASR-1.7B` 会加载两次，占用两份 GPU 显存。

## 功能特性

- 单进程、单模型实例：`Qwen3-ASR-1.7B` 只加载一次。
- 离线长音频：复用 VAD 分段，默认目标段长 `45s`，最大单段 `60s`。
- 在线实时转写：WebSocket 收音频、返回 `ack` / `partial` / `final`。
- 长音频在线策略：推荐 `120s window + 0s overlap + 顺序 WebSocket session`。
- 背压保护：服务端接收、推理、发送解耦，客户端验证脚本支持 inflight window。
- Forced Aligner：默认禁用；可配置远端 forced aligner 服务。

## 环境要求

- Linux + NVIDIA GPU。
- 推荐显存：L20 级别或更高。
- 推荐 Python：`3.11` / `3.12` / `3.13`。
- 当前真实验证环境：`conda activate qwen-asr`，Python `3.13.5`，NVIDIA L20。

如果 `vllm==0.14.0` 在某个 Python 版本没有可用 wheel，请优先创建 Python 3.11 或 3.12 环境。

## 安装依赖

```bash
cd /workspace/project/Qwen3-ASR-Toolkit
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen-asr

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

安装完成后应能看到正式命令：

```bash
which qwen3-asr-native-server
qwen3-asr-native-server --help
```

## 下载模型

模型文件不提交到 Git。请下载到 `models/` 目录：

```bash
pip install modelscope

modelscope download \
  --model 'Qwen/Qwen3-ASR-1.7B' \
  --local_dir '/workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ASR-1.7B'

modelscope download \
  --model 'Qwen/Qwen3-ForcedAligner-0.6B' \
  --local_dir '/workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ForcedAligner-0.6B'
```

离线加载推荐环境变量：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN3_ASR_MODEL_PATH=/workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ASR-1.7B
export QWEN3_ALIGNER_MODEL_PATH=/workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ForcedAligner-0.6B
```

## 启动统一 Native 服务

推荐使用正式命令：

```bash
qwen3-asr-native-server \
  --host 0.0.0.0 \
  --port 10012 \
  --model-path /workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.8 \
  --max-new-tokens 128 \
  --chunk-size-sec 1.0 \
  --unfixed-chunk-num 2 \
  --unfixed-token-num 5 \
  --audio-queue-size 8 \
  --send-queue-size 32 \
  --decode-timeout-sec 0 \
  --enable-offline-api \
  --offline-num-threads 1 \
  --vad-target-segment-s 45 \
  --vad-max-segment-s 60 \
  --max-concurrent-asr-jobs 1 \
  --aligner-mode disabled
```

也可以使用脚本入口：

```bash
bash scripts/run_native_server.sh
```

或兼容方式：

```bash
python deploy/vllm_streaming_server_native.py --model-path models/Qwen3-ASR-1.7B --port 10012
```

## 健康检查

```bash
curl -s http://127.0.0.1:10012/health
```

关键字段：

```json
{
  "status": "ok",
  "capabilities": {
    "offline_http": true,
    "native_websocket": true,
    "forced_aligner": "disabled",
    "max_concurrent_asr_jobs": 1,
    "asr_model_loaded_once": true
  }
}
```

## 离线 HTTP 转写

```bash
curl -X POST "http://127.0.0.1:10012/api/v1/offline/transcribe" \
  -F "audio_file=@sample/sample_0.mp3" \
  -F "context=" \
  -F "use_forced_aligner=false"
```

长音频也使用同一接口。服务端会自动加载音频、重采样、VAD 分段，并保证单段不超过 `vad_max_segment_s`。

## 在线 WebSocket 转写

WebSocket 地址：

```text
ws://127.0.0.1:10012/ws/stream
```

协议：

1. 客户端发送 `start` JSON。
2. 客户端发送 `float32le PCM mono 16k` 二进制音频 chunk。
3. 服务端返回 `ack` 和 `partial`。
4. 客户端发送 `finish`。
5. 服务端返回 `final`。

单窗口验证：

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

## 长音频在线转写策略

单个 WebSocket session 推荐不超过 `120s`。更长音频使用 windowed 方式：

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

注意：当前在线方案是服务层 windowed streaming，不声称是 KV-cache 级 true incremental streaming。

## Forced Aligner

默认：

```bash
--aligner-mode disabled
```

此模式不会加载 `Qwen3-ForcedAligner-0.6B`。如果请求 `use_forced_aligner=true`，接口会返回 forced-aligner metadata，说明当前服务未启用对齐。

如需启用远端 forced aligner，可使用：

```bash
--aligner-mode remote \
--aligner-base-url http://127.0.0.1:10013 \
--aligner-api-key EMPTY \
--aligner-model-path Qwen3-ForcedAligner-0.6B
```

## 目录结构

```text
Qwen3-ASR-Toolkit/
├── deploy/                 # Native server 和验证脚本
├── doc/                    # 中文文档
├── qwen3_asr_toolkit/       # 音频、离线分段、forced aligner 工具
├── qwen_asr/                # 内嵌 Qwen3-ASR 模型实现和 vLLM 插件
├── sample/                  # 样例音频和参考文本
├── scripts/                 # 启动脚本
└── tests/                   # 单元测试
```

## 不推荐使用的旧方式

本项目不再推荐以下方式：

```bash
vllm serve models/Qwen3-ASR-1.7B
OPENAI_BASE_URL=http://... qwen3-asr-server
qwen3-asr-offline-cli
qwen3-asr-stream-cli
```

原因：这些旧路径会引入额外服务进程或旧协议，容易造成 `Qwen3-ASR-1.7B` 重复加载。

## 更多文档

- `doc/MODEL_DOWNLOAD.md`：模型下载。
- `doc/DEPLOYMENT.md`：部署说明。
- `doc/API.md`：HTTP 和 WebSocket API。
- `doc/ARCHITECTURE.md`：架构说明。
- `doc/VALIDATION.md`：验证命令和验收标准。
