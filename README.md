# Qwen3-ASR-Toolkit

一个面向云端部署的 Qwen3-ASR 工具集，支持：

- 离线语音转写：REST API + CLI
- 实时语音转写：WebSocket + CLI
- 长音频自动 VAD 切分，且单段严格不超过 `60s`
- 可选接入 `Qwen3-ForcedAligner-0.6B` 做离线时间戳对齐
- 基于 vLLM OpenAI 兼容接口调用 ASR 服务

当前仓库已经内置了从上游迁出的 `qwen_asr/` 包和示例代码，不再依赖已删除的 `references/` 目录。

## 文档导航

- 文档总索引：[doc/README.md](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/doc/README.md)
- 详细部署文档：[doc/DEPLOYMENT.md](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/doc/DEPLOYMENT.md)
- 使用说明文档：[doc/USAGE.md](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/doc/USAGE.md)
- 开发者指南：[doc/DEVELOPER_GUIDE.md](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/doc/DEVELOPER_GUIDE.md)
- 技术改进说明：[doc/TECH_IMPROVEMENTS.md](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/doc/TECH_IMPROVEMENTS.md)
- 云端 API/WS 协议说明：[doc/cloud_api_ws_usage.md](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/doc/cloud_api_ws_usage.md)
- vLLM 迁移记录：[doc/vllm_migration.md](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/doc/vllm_migration.md)

## 核心能力

- 离线转写接口：`POST /api/v1/offline/transcribe`
- 实时转写接口：`ws://<host>:<port>/ws/v1/realtime/transcribe`
- 离线转写支持参数 `use_forced_aligner=true|false`
- 实时转写不做对齐，只做准实时增量转写
- `sample/` 目录内置真实测试音频

## 当前项目结构

```text
qwen3_asr_toolkit/    主工具实现（服务端、CLI、离线/实时转写、对齐客户端）
qwen_asr/             从上游迁入的官方包
doc/                  中文文档
scripts/              部署与验证脚本
sample/               真实测试音频
tests/                单元测试
examples/             上游示例快照
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### 2. 配置 `.env`

项目根目录的 [`.env.example`](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/.env.example) 已给出模板。

当前已验证可用的配置格式如下：

```env
OPENAI_API_KEY=your_asr_api_key
OPENAI_BASE_URL=http://your-asr-host:10010/v1
QWEN3_ASR_MODEL=Qwen3-ASR-1.7B

QWEN3_ALIGNER_BASE_URL=http://your-aligner-host:10011
QWEN3_ALIGNER_API_KEY=your_aligner_api_key
QWEN3_ALIGNER_MODEL=Qwen3-ForcedAligner-0.6B

QWEN3_ASR_NUM_THREADS=4
QWEN3_ASR_VAD_TARGET_SEGMENT_S=45
QWEN3_ASR_VAD_MAX_SEGMENT_S=60
QWEN3_ASR_STREAM_DECODE_INTERVAL_MS=600
QWEN3_ASR_STREAM_MIN_CHUNK_MS=200
QWEN3_ASR_STREAM_FINALIZE_SILENCE_MS=600
QWEN3_ASR_STREAM_MAX_SEGMENT_S=20
QWEN3_ALIGNER_TIMEOUT_S=120
QWEN3_ALIGNER_TIMESTAMP_SEGMENT_TIME_MS=80
```

### 3. 启动服务

```bash
python -m qwen3_asr_toolkit.server --host 0.0.0.0 --port 18000
```

健康检查：

```bash
curl http://127.0.0.1:18000/health
```

### 4. 运行离线 CLI

```bash
python -m qwen3_asr_toolkit.offline_cli \
  -i sample/sample_0.mp3 \
  -u http://127.0.0.1:18000/api/v1/offline/transcribe \
  --use-forced-aligner \
  --save-text
```

### 5. 运行实时 CLI

```bash
python -m qwen3_asr_toolkit.realtime_cli \
  -i sample/sample_0.mp3 \
  -u ws://127.0.0.1:18000/ws/v1/realtime/transcribe \
  --chunk-ms 300 \
  --decode-interval-ms 600 \
  --finalize-silence-ms 600
```

## 部署脚本

已提供部署脚本：

- [scripts/deploy_server.sh](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/scripts/deploy_server.sh)
- [scripts/run_sample_tests.sh](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/scripts/run_sample_tests.sh)
- `systemd` 示例：[deploy/systemd/qwen3-asr-toolkit.service](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/deploy/systemd/qwen3-asr-toolkit.service)

常用命令：

```bash
bash scripts/deploy_server.sh install
bash scripts/deploy_server.sh start
bash scripts/deploy_server.sh status
bash scripts/deploy_server.sh logs
```

## 实测结果摘要

基于 `sample/` 中真实文件已完成联调验证：

- `sample/sample_0.mp3`
  - 离线转写正常
  - 开启对齐后可返回词级时间戳
- `sample/deutsch.mp3`
  - 长音频会自动切分
  - 最大切分长度小于 `60s`
- `sample/sample_2.m4a`
  - 可完成长音频离线转写
- WebSocket 实时链路已完成增量转写验证

## 说明

- 如果只做离线/实时转写，不需要安装 `qwen_asr` 的完整额外依赖。
- 如果要运行仓库内嵌的上游 `qwen_asr` 示例，可以安装：
  - `pip install -e ".[qwen-asr]"`
  - `pip install -e ".[qwen-asr-vllm]"`

## 许可证

- 本项目主代码遵循仓库根目录 [LICENSE](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/LICENSE)
- 迁入的 `qwen_asr/` 与上游示例保留其原始许可证说明
