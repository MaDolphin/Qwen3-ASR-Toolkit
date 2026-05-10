# 模型下载

本项目默认从本地 `models/` 目录加载模型。模型文件不提交到 Git。

## 安装 ModelScope

```bash
pip install modelscope
```

## 下载 Qwen3-ASR-1.7B

```bash
modelscope download \
  --model 'Qwen/Qwen3-ASR-1.7B' \
  --local_dir '/workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ASR-1.7B'
```

## 下载 Qwen3-ForcedAligner-0.6B

```bash
modelscope download \
  --model 'Qwen/Qwen3-ForcedAligner-0.6B' \
  --local_dir '/workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ForcedAligner-0.6B'
```

## 推荐环境变量

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN3_ASR_MODEL_PATH=/workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ASR-1.7B
export QWEN3_ALIGNER_MODEL_PATH=/workspace/project/Qwen3-ASR-Toolkit/models/Qwen3-ForcedAligner-0.6B
```

## Git 规则

`.gitignore` 已忽略：

```text
models/
runtime/
```

不要把模型权重、运行日志、验证报告提交到仓库。
