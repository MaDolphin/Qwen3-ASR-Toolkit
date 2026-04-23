# 部署文档

本文档面向准备把 `Qwen3-ASR-Toolkit` 部署到云服务器的使用者，覆盖环境准备、依赖安装、配置项说明、启动方式、脚本使用和常见排障。

## 1. 部署目标

部署完成后，服务会提供两类入口：

- 离线转写 REST API：`POST /api/v1/offline/transcribe`
- 实时转写 WebSocket：`/ws/v1/realtime/transcribe`

依赖的后端模型服务：

- `Qwen3-ASR-1.7B`：通过 vLLM OpenAI 兼容接口提供转写能力
- `Qwen3-ForcedAligner-0.6B`：通过 vLLM `/pooling` + `/tokenize` 提供离线时间戳对齐能力

## 2. 服务器要求

建议环境：

- 操作系统：Ubuntu 20.04 / 22.04
- Python：`3.10+`
- FFmpeg：已安装并在 `PATH` 中可用
- 网络：可访问你的 ASR vLLM 服务和 Forced Aligner 服务

当前项目本身不直接加载大模型，只调用远端 vLLM，所以这台应用服务器不需要大显存 GPU。

## 3. 目录约定

假设部署目录为：

```bash
/srv/qwen3-asr-toolkit
```

进入目录：

```bash
cd /srv/qwen3-asr-toolkit
```

## 4. 安装依赖

### 4.1 系统依赖

Ubuntu：

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg
```

### 4.2 Python 依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## 5. 配置 `.env`

项目会自动读取根目录 `.env`。

可以直接参考 [`.env.example`](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/.env.example)。

典型配置如下：

```env
OPENAI_API_KEY=your_asr_api_key
OPENAI_BASE_URL=http://172.28.245.150:10010/v1
QWEN3_ASR_MODEL=Qwen3-ASR-1.7B

QWEN3_ALIGNER_BASE_URL=http://172.28.245.150:10011
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

关键说明：

- `OPENAI_BASE_URL`
  - 需要带 `/v1`
  - 用于 Qwen3-ASR 主模型
- `QWEN3_ALIGNER_BASE_URL`
  - 当前对齐模型不是 `chat/completions`
  - 服务端会调用 `/pooling` 与 `/tokenize`
  - 这里不要加 `/v1`
- `QWEN3_ASR_VAD_MAX_SEGMENT_S`
  - 建议保持 `60`
  - 项目保证单段不会超过这个值
- `QWEN3_ASR_STREAM_DECODE_INTERVAL_MS`
  - 实时模式多久返回一次增量结果
- `QWEN3_ASR_STREAM_FINALIZE_SILENCE_MS`
  - 实时模式检测到多长静音后提交当前语音段
- `QWEN3_ASR_STREAM_MAX_SEGMENT_S`
  - 当前语音段持续过长时强制提交的上限

## 6. 启动服务

### 6.1 前台启动

```bash
source .venv/bin/activate
python -m qwen3_asr_toolkit.server --host 0.0.0.0 --port 18000
```

### 6.2 使用部署脚本启动

推荐使用脚本：

```bash
bash scripts/deploy_server.sh install
bash scripts/deploy_server.sh start
```

脚本会：

- 自动创建 `.venv`
- 安装依赖
- 启动服务到后台
- 将 PID 写入 `runtime/server.pid`
- 将日志写入 `runtime/server.log`

常用命令：

```bash
bash scripts/deploy_server.sh start
bash scripts/deploy_server.sh stop
bash scripts/deploy_server.sh restart
bash scripts/deploy_server.sh status
bash scripts/deploy_server.sh logs
```

## 7. 部署后验证

### 7.1 健康检查

```bash
curl http://127.0.0.1:18000/health
```

预期返回：

```json
{
  "status": "ok",
  "model": "Qwen3-ASR-1.7B",
  "sample_rate": 16000,
  "vad_max_segment_s": 60,
  "stream_decode_interval_ms": 600,
  "stream_min_chunk_ms": 200,
  "stream_finalize_silence_ms": 600,
  "stream_max_segment_sec": 20.0
}
```

### 7.2 样例验证

```bash
bash scripts/run_sample_tests.sh
```

它会验证：

- 短音频离线转写
- 短音频离线对齐
- 长音频离线切分约束
- 实时 WebSocket 转写链路

## 8. 生产部署建议

当前推荐的默认路径是：

- 直接启动 `qwen3_asr_toolkit.server`
- 用 `systemd` 或 `supervisor` 托管
- 通过云服务器安全组或防火墙控制访问来源

### 8.1 进程托管

长期运行建议用 `systemd` 或 `supervisor` 托管，而不是只靠 `nohup`。

仓库内已提供 `systemd` 示例文件：

- [deploy/systemd/qwen3-asr-toolkit.service](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/deploy/systemd/qwen3-asr-toolkit.service)

使用方式：

```bash
sudo cp deploy/systemd/qwen3-asr-toolkit.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable qwen3-asr-toolkit
sudo systemctl start qwen3-asr-toolkit
sudo systemctl status qwen3-asr-toolkit
```

注意按你的实际环境修改：

- `User`
- `Group`
- `WorkingDirectory`
- `ExecStart`

## 9. 常见问题

### 9.1 `chat/completions` 对 aligner 返回 404

这是正常的。

`Qwen3-ForcedAligner-0.6B` 当前服务不是走 `chat/completions`，而是：

- `/tokenize`
- `/pooling`

项目已经按这个方式适配。

### 9.2 WebSocket 客户端连不上本地地址

如果机器上有代理环境，可能需要临时关闭：

```bash
ALL_PROXY= \
HTTP_PROXY= \
HTTPS_PROXY= \
NO_PROXY=127.0.0.1,localhost \
python -m qwen3_asr_toolkit.realtime_cli ...
```

### 9.3 长音频很慢

这是正常现象，主要耗时来自：

- VAD 切分
- 多段 ASR 调用
- 可选的 forced alignment

如果对齐不是必须，不要传 `use_forced_aligner=true`。

### 9.4 实时模式为什么没有时间戳

实时模式当前只做增量转写，不做对齐。

这是产品边界，不是 bug。
