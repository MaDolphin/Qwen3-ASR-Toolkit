# 部署说明

## 1. 环境准备

```bash
cd /workspace/project/Qwen3-ASR-Toolkit
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen-asr
python -V
```

推荐 Python `3.11` / `3.12` / `3.13`。

## 2. 安装依赖

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

验证命令入口：

```bash
which qwen3-asr-native-server
qwen3-asr-native-server --help
```

## 3. 下载模型

详见 `docs/MODEL_DOWNLOAD.md`。

## 4. 启动服务

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN3_ASR_MODEL_PATH=/workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ASR-1.7B

qwen3-asr-native-server \
  --host 0.0.0.0 \
  --port 10012 \
  --model-path /workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.50 \
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
  --aligner-mode remote
```

脚本方式：

```bash
bash scripts/run_native_server.sh
```

## 5. 健康检查

```bash
curl -s http://127.0.0.1:10012/health
```

## 6. GPU 检查

```bash
nvidia-smi
ps -ef | grep -E "qwen3-asr-native-server|vllm_streaming_server_native|VLLM::EngineCore|vllm serve"
```

通过标准：

- 只有一个 Native server 进程。
- 只有一个 `VLLM::EngineCore`。
- 没有额外 `vllm serve models/Qwen3-ASR-1.7B`。

## 7. systemd 部署

示例文件：

```text
deploy/systemd/qwen3-asr-native-server.service
```

安装：

```bash
sudo cp deploy/systemd/qwen3-asr-native-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable qwen3-asr-native-server
sudo systemctl start qwen3-asr-native-server
sudo systemctl status qwen3-asr-native-server
```

查看日志：

```bash
journalctl -u qwen3-asr-native-server -f
```

## 一键部署两个模型

```bash
bash scripts/deploy_native_asr.sh
```

该脚本会启动两个服务：

| 服务 | 地址 | 模型 |
| --- | --- | --- |
| ASR Native Server | `http://127.0.0.1:10012` | `models/Qwen3-ASR-1.7B` |
| ForcedAligner vLLM Server | `http://127.0.0.1:10013` | `models/Qwen3-ForcedAligner-0.6B` |

默认低显存参数：

```text
ASR_GPU_MEMORY_UTILIZATION=0.50
ASR_KV_CACHE_MEMORY_BYTES=8G
ALIGNER_GPU_MEMORY_UTILIZATION=0.10
ALIGNER_KV_CACHE_MEMORY_BYTES=2G
```

停止服务：

```bash
bash scripts/stop_native_asr.sh
```

CLI 验证：

```bash
qwen3-asr-cli health
qwen3-asr-offline-cli --input-file sample/sample_0.mp3
qwen3-asr-stream-cli --input-file sample/sample_2.m4a --duration-sec 120 --realtime
```

## ForcedAligner 启动参数

一键部署脚本使用 vLLM pooling 模式启动 ForcedAligner，关键参数如下：

```bash
--runner pooling \
  --pooler-config '{"task":"token_classify"}' \
  --hf-overrides '{"architectures":["Qwen3ASRForcedAlignerForTokenClassification"]}' \
  --gpu-memory-utilization 0.10
```
