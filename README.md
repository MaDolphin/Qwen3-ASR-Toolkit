# Qwen3-ASR-Toolkit

Qwen3-ASR-Toolkit 是一个面向生产部署的 Qwen3-ASR 语音识别服务项目，提供统一的 Native ASR 服务、离线 HTTP 转写、在线 WebSocket 转写、命令行客户端以及可选 ForcedAligner 时间戳能力。

## 核心能力

- **统一 ASR 服务**：`Qwen3-ASR-1.7B` 在 Native server 进程中加载一次。
- **离线转写**：`POST /api/v1/offline/transcribe` 支持长音频 VAD 分段，默认最大单段 `60s`。
- **在线转写**：`WS /ws/stream` 支持实时发送音频、返回 `ack`、`partial`、`final`。
- **命令行客户端**：提供 `qwen3-asr-offline-cli`、`qwen3-asr-stream-cli`、`qwen3-asr-cli`。
- **ForcedAligner**：可自动部署 `Qwen3-ForcedAligner-0.6B` 作为远端 pooling 服务，为离线结果提供时间戳。
- **低显存部署**：默认参数面向已占用约 `11GB` 显存的 L20，给后续模型保留显存空间。

## 架构概览

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

## 环境要求

- Linux + NVIDIA GPU。
- 推荐 GPU：L20 或更高。
- 推荐 Python：`3.11` / `3.12` / `3.13`。
- 当前验证环境：`conda activate qwen-asr`，Python `3.13.5`，NVIDIA L20。

## .env 配置

部署配置统一来自 `.env`。首次运行 `scripts/deploy_native_asr.sh` 时，如果 `.env` 不存在，脚本会自动从 `.env.example` 复制生成。

常用配置：

```bash
QWEN3_CUDA_VISIBLE_DEVICES=0
QWEN3_ASR_PORT=10012
QWEN3_ALIGNER_PORT=10013
QWEN3_GRADIO_PORT=7860
QWEN3_ASR_MAX_CONCURRENT_JOBS=1
QWEN3_ASR_OFFLINE_NUM_THREADS=2
```

多卡机器默认 ASR 和 ForcedAligner 使用同一张卡：

```bash
QWEN3_CUDA_VISIBLE_DEVICES=1
```

高级分卡配置：

```bash
QWEN3_ASR_CUDA_VISIBLE_DEVICES=0
QWEN3_ALIGNER_CUDA_VISIBLE_DEVICES=1
```

## 多人访问与并发

服务支持多人同时接入 HTTP 离线转写和 WebSocket 实时转写。当前 ASR 模型推理采用单 worker 串行执行，超过后会在服务端排队，不会因为超过并发槽就直接拒绝连接。

实际可承载人数取决于任务类型、音频长度、GPU 显存、KV cache 和实时延迟要求：

- 短音频离线：适合多人同时提交，超过并发槽后排队。
- 长音频离线：会被切成多个 segment，可能占用多个并发槽。
- 实时 WebSocket：每个连接持续产生 chunk 推理请求，多路实时会共享并发槽。

当前采用单模型单 worker 串行推理，WebSocket 实时任务优先于离线分段任务。该模式避免同一个 vLLM LLM 实例并行 generate 触发 EngineCore 异常；如离线长音频影响实时延迟，请将 `QWEN3_ASR_OFFLINE_NUM_THREADS` 降到 `1`。

调度策略：

```text
HTTP 离线分段 -> ASR priority queue(priority=10)
WebSocket chunk/final -> ASR priority queue(priority=0)
                    -> single ASR worker -> shared Qwen3-ASR-1.7B
```

`QWEN3_ASR_MAX_CONCURRENT_JOBS` 现在保留为兼容参数，实际模型推理固定串行；`/health.capabilities.requested_max_concurrent_asr_jobs` 会记录用户请求值，`max_concurrent_asr_jobs` 固定为 `1`。

## 一键部署

一键部署会自动检查依赖、下载模型并启动三个进程：ForcedAligner、ASR Native Server 和 Gradio 客户端服务：

```bash
bash scripts/deploy_native_asr.sh
```

使用当前已激活环境部署：

```bash
conda activate qwen-asr
bash scripts/deploy_native_asr.sh --no-conda-activate
```

默认服务地址：

```text
ASR local:      http://127.0.0.1:10012
ASR remote:     http://服务器IP:10012
WS local:       ws://127.0.0.1:10012/ws/stream
WS remote:      ws://服务器IP:10012/ws/stream
Aligner local:  http://127.0.0.1:10013
Aligner remote: http://服务器IP:10013
Gradio local:  http://127.0.0.1:7860
Gradio remote: http://服务器IP:7860
```

Health 中应看到：

```text
capabilities.asr_scheduler=priority-single-worker
capabilities.realtime_priority=true
capabilities.max_concurrent_asr_jobs=1
```

默认低显存参数：

```text
ASR_GPU_MEMORY_UTILIZATION=0.30
ASR_KV_CACHE_MEMORY_BYTES=8G
ASR_CPU_OFFLOAD_GB=0
ASR_MAX_MODEL_LEN=65536

ALIGNER_GPU_MEMORY_UTILIZATION=0.10
ALIGNER_KV_CACHE_MEMORY_BYTES=2G
# ForcedAligner 使用 vLLM pooling token_classify，内部脚本会设置 --pooler-config 与 --hf-overrides。
ALIGNER_CPU_OFFLOAD_GB=0
```

停止服务：

```bash
bash scripts/stop_native_asr.sh
```

## 日志目录

一键部署会创建 `logs/`，用于排查线上问题：

```text
logs/asr_server.log      # ASR Native Server 访问和运行日志
logs/gradio_server.log   # Gradio 页面访问和运行日志
logs/aligner_server.log  # ForcedAligner 运行日志
```

测试完成后请执行 `bash scripts/stop_native_asr.sh` 停止服务，避免占用 GPU；也可以使用 `STOP_AFTER_TEST=1 bash scripts/test_native_asr_functional.sh` 让测试脚本结束时自动停止服务。

## 手动安装

```bash
cd /workspace/project/Qwen3-ASR-Toolkit
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen-asr

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

开发测试依赖：

```bash
python -m pip install -r requirements-dev.txt
```

## 模型下载

```bash
pip install modelscope

modelscope download \
  --model 'Qwen/Qwen3-ASR-1.7B' \
  --local_dir '/workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ASR-1.7B'

modelscope download \
  --model 'Qwen/Qwen3-ForcedAligner-0.6B' \
  --local_dir '/workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ForcedAligner-0.6B'
```

## 启动 ASR 服务

如果 ForcedAligner 已经在 `127.0.0.1:10013` 启动，可以单独启动 ASR：

```bash
export CUDA_VISIBLE_DEVICES=0

qwen3-asr-native-server \
  --host 0.0.0.0 \
  --port 10012 \
  --model-path /workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.30 \
  --kv-cache-memory-bytes 8G \
  --max-model-len 65536 \
  --max-new-tokens 128 \
  --enable-offline-api \
  --offline-num-threads 2 \
  --max-concurrent-asr-jobs 1 \
  --aligner-mode remote \
  --aligner-base-url http://127.0.0.1:10013 \
  --aligner-api-key EMPTY \
  --aligner-model-path Qwen3-ForcedAligner-0.6B
```

## CLI 使用方式

### 健康检查

```bash
qwen3-asr-cli health
qwen3-asr-cli health --server http://服务器IP:10012
```

### 离线转写

```bash
qwen3-asr-offline-cli \
  --server http://服务器IP:10012 \
  --input-file sample/sample_0.mp3 \
  --output-json runtime/cli_sample_0.json \
  --output-text runtime/cli_sample_0.txt
```

### 离线转写 + ForcedAligner

```bash
qwen3-asr-offline-cli \
  --server http://服务器IP:10012 \
  --input-file sample/sample_0.mp3 \
  --use-forced-aligner \
  --output-json runtime/cli_sample_0_aligned.json
```

### 在线 WebSocket 转写

```bash
qwen3-asr-stream-cli \
  --server http://服务器IP:10012 \
  --input-file sample/sample_2.m4a \
  --duration-sec 120 \
  --output-json runtime/cli_sample_2_120s.json \
  --realtime
```

### 统一 CLI 入口

```bash
qwen3-asr-cli offline --input-file sample/sample_0.mp3
qwen3-asr-cli stream --input-file sample/sample_2.m4a --duration-sec 120
qwen3-asr-cli health --server http://服务器IP:10012
```

## Gradio Web UI

Gradio 只作为客户端访问已部署的 Native ASR 服务，不加载 ASR 模型。默认监听 `0.0.0.0:7860`，方便远程浏览器访问。

安装客户端依赖：

```bash
pip install -r requirements-client.txt
# 或者
pip install -e .[client]
```

启动 Web UI：

```bash
qwen3-asr-gradio \
  --server http://服务器IP:10012 \
  --host 0.0.0.0 \
  --port 7860
```

功能：

- 离线 Tab：上传音频文件，调用 `POST /api/v1/offline/transcribe`。
- 实时 Tab：浏览器授权麦克风，Gradio 后端连接 `WS /ws/stream`，持续展示 partial，停止后展示 final。

## Python examples

```bash
python examples/offline_http_example.py \
  --server http://127.0.0.1:10012 \
  --input sample/sample_0.mp3 \
  --output-json runtime/examples/offline_result.json

python examples/realtime_ws_example.py \
  --server http://127.0.0.1:10012 \
  --input sample/sample_2.m4a \
  --duration-sec 30 \
  --realtime \
  --output-json runtime/examples/ws_result.json

python examples/windowed_long_audio_example.py \
  --uri ws://127.0.0.1:10012/ws/stream \
  --input sample/sample_2.m4a \
  --output runtime/examples/windowed_result.json \
  --case-dir runtime/examples/windowed_cases \
  --window-sec 120 \
  --chunk-ms 500
```

## deploy 目录说明

`deploy/` 只保留服务端部署相关文件：

| 文件 | 用途 |
| --- | --- |
| `deploy/vllm_streaming_server_native.py` | 统一 Native ASR server，提供 health、离线 HTTP 和在线 WebSocket。 |
| `deploy/forced_aligner_server.py` | ForcedAligner vLLM pooling server 启动封装。 |
| `deploy/streaming_utils.py` | Native server 使用的流式转写辅助函数。 |
| `deploy/systemd/` | 生产 systemd 服务文件。 |

验证脚本已移动到 `examples/validation/`。

## HTTP API 使用方式

本机访问使用 `127.0.0.1`，远程访问将地址替换为 `服务器IP:10012`：

```bash
curl -X POST "http://127.0.0.1:10012/api/v1/offline/transcribe" \
  -F "audio_file=@sample/sample_0.mp3" \
  -F "context=" \
  -F "use_forced_aligner=false"
```

## WebSocket API 使用方式

WebSocket 地址：

```text
本机访问：ws://127.0.0.1:10012/ws/stream
远程访问：ws://服务器IP:10012/ws/stream
```

客户端发送 `start`，再发送 `float32le PCM mono 16k` 音频 chunk，最后发送 `finish`。服务端返回 `started`、`ack`、`partial`、`final` 或 `error`。

## 长音频在线策略

单个 WebSocket session 推荐不超过 `120s`。更长音频使用 windowed 方式：

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

## 功能测试

完整功能测试命令：

```bash
bash scripts/test_native_asr_functional.sh
```

测试报告输出到：

```text
runtime/native_deploy_validation/FUNCTIONAL_TEST_REPORT.md
```

当前 L20 环境已完成真实功能验证：

| 测试项 | 结果 | 关键指标 |
| --- | --- | --- |
| Health | 通过 | ASR `status=ok`，ForcedAligner 可访问 |
| 离线短音频 | 通过 | `sample_0.mp3` 文本非空，`segments=1` |
| 离线 ForcedAligner | 通过 | `available=true`，返回 token 时间戳条目 |
| 离线长音频 | 通过 | `sample_2.m4a` 文本非空，VAD 分段完成 |
| WebSocket 120s harness | 通过 | `chunks=240`，`ack=240`，`partials=239`，`final` 非空 |
| WebSocket 120s CLI | 通过 | `chunks=240`，`ack=240`，`partials=239`，`final` 非空 |
| CLI 远程参数 | 通过 | `--server http://服务器IP:10012` 可自动推导 HTTP/WS 地址 |

本次低显存参数实测显存参考：

```text
启动前：4 MiB used
ForcedAligner 启动后：4635 MiB used
ASR 启动后：19350 MiB used
完整功能测试后：26004 MiB used
```

说明：显存占用会受到 vLLM KV cache、CUDA graph、请求峰值和其他进程影响，生产部署时请以 `nvidia-smi` 实测为准。

## 目录结构

```text
Qwen3-ASR-Toolkit/
├── client/                 # CLI 和 Gradio 客户端
├── deploy/                 # Native server、ForcedAligner server 和 systemd
├── docs/                   # 中文文档
├── examples/               # Python 调用示例和验证脚本
├── qwen3_asr_toolkit/       # 音频、离线分段、ForcedAligner 客户端
├── qwen_asr/                # 内嵌 Qwen3-ASR 模型实现和 vLLM 插件
├── sample/                  # 样例音频和参考文本
├── scripts/                 # 部署、停止、测试脚本
└── tests/                   # 单元测试
```

## 更多文档

- `docs/MODEL_DOWNLOAD.md`：模型下载。
- `docs/DEPLOYMENT.md`：部署说明。
- `docs/API.md`：HTTP、WebSocket 和 CLI 对应关系。
- `docs/ARCHITECTURE.md`：架构说明。
- `docs/VALIDATION.md`：验证命令和验收标准。
