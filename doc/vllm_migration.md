# Qwen3-ASR-Toolkit：DashScope → vLLM 自部署迁移文档

> **日期**: 2026-04-16  
> **背景**: 将 Qwen3-ASR 模型从 DashScope 云端 API 迁移至 vLLM 0.19.0 自部署方案  
> **部署地址**: `172.28.245.150:10010`  
> **模型名称**: `Qwen3-ASR-1.7B`（通过 `--served-model-name` 指定）

---

## 1. 迁移概述

原项目使用阿里云 DashScope SDK 调用 Qwen3-ASR-Flash 在线 API。本次修改将其切换为通过 **OpenAI 兼容 API** 调用 **vLLM 自部署的 Qwen3-ASR-1.7B** 模型。

### 核心变化

| 维度 | 迁移前（DashScope） | 迁移后（vLLM） |
|---|---|---|
| SDK | `dashscope` | `openai` |
| API 格式 | `dashscope.MultiModalConversation.call()` | `client.chat.completions.create()` |
| 音频传输 | `file://` 本地路径 | base64 编码 Data URI |
| 认证方式 | `DASHSCOPE_API_KEY` | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |
| 响应格式 | `annotations.language` 字段 | 文本内嵌 `language XXX<asr_text>...` |
| 音频长度限制 | DashScope 3 分钟限制 | vLLM encoder cache 限制（默认 2048 tokens ≈ 150s） |

---

## 2. 修改文件清单

### 2.1 `qwen3_asr_toolkit/qwen3asr.py`

**主要变更**：

```diff
- import dashscope
+ import base64
+ import re
+ from openai import OpenAI
```

- **`QwenASR.__init__`**：新增 `base_url` 和 `api_key` 参数，内部创建 `OpenAI` client
- **`_encode_audio_base64()`**：新增方法，将本地音频文件编码为 `data:<mime>;base64,...` 格式的 Data URI
- **`asr()` 方法**：
  - 音频传输：`file://` 路径 → base64 Data URI
  - API 调用：`dashscope.MultiModalConversation.call()` → `client.chat.completions.create()`
  - 消息格式：使用 OpenAI 兼容的 `audio_url` 类型
  - 响应解析：正则提取 `language XXX<asr_text>转写文本` 格式
  - 错误处理：encoder cache 超限时立即抛出，不再无意义重试

**消息格式对比**：

```python
# 迁移前（DashScope）
messages = [
    {"role": "system", "content": [{"text": context}]},
    {"role": "user",   "content": [{"audio": "file:///path/to/audio.wav"}]}
]
response = dashscope.MultiModalConversation.call(
    model=self.model, messages=messages, result_format="message",
    asr_options={"enable_lid": True, "enable_itn": False}
)

# 迁移后（vLLM OpenAI 兼容）
messages = [
    {"role": "system", "content": [{"type": "text", "text": context}]},
    {"role": "user",   "content": [{"type": "audio_url", "audio_url": {"url": data_uri}}]}
]
response = self.client.chat.completions.create(
    model=self.model, messages=messages
)
```

**响应解析对比**：

```python
# 迁移前：语言信息在 annotations 字段
lang_code = output["message"]["annotations"][0]["language"]  # "zh"
recog_text = output["message"]["content"][0]["text"]

# 迁移后：语言信息嵌在文本中
# 原始输出: "language Chinese<asr_text>这是转写文本"
match = re.match(r'^language\s+(\w+)\s*<asr_text>(.*)', raw_text, re.DOTALL)
language = match.group(1)   # "Chinese"
recog_text = match.group(2) # "这是转写文本"
```

### 2.2 `qwen3_asr_toolkit/call_api.py`

**命令行参数变更**：

| 迁移前 | 迁移后 | 说明 |
|---|---|---|
| `--dashscope-api-key` / `-key` | `--api-key` / `-key` | API Key |
| *(无)* | `--base-url` / `-url` | vLLM 服务地址，默认 `http://172.28.245.150:10010/v1` |
| *(无)* | `--model` / `-m` | 模型名称，默认自动检测 |
| `--vad-segment-threshold` 默认 `120` | 默认 `50` | 适配 vLLM encoder cache 限制 |

**逻辑变更**：

- **模型自动检测**：未指定 `--model` 时，通过 `client.models.list()` 自动获取 vLLM 上部署的模型名
- **VAD 分段触发**：阈值从固定 `180s` 改为与 `--vad-segment-threshold`（默认 50s）一致，确保音频不会超过 vLLM encoder cache 限制
- **移除 `dashscope` 依赖**：不再 `import dashscope` 或设置 `dashscope.api_key`

### 2.3 `requirements.txt`

```diff
- dashscope
+ openai
```

### 2.4 `setup.py`

```diff
  install_requires=[
-     'dashscope',
+     'openai',
      ...
  ]
```

---

## 3. 部署配置

### 3.1 vLLM 服务端

当前服务启动参数（含 `--served-model-name`）：

```bash
vllm serve Qwen/Qwen3-ASR-1.7B \
    --served-model-name Qwen3-ASR-1.7B \
    --port 10010 \
    --api-key <your_api_key>
```

> **注意**：服务端默认 `--limit-mm-per-prompt` 对应 encoder cache 大小为 2048 tokens（约 150s 音频）。如需处理更长的单段音频，需增大此参数，例如：
>
> ```bash
> --limit-mm-per-prompt '{"audio": 4}'
> ```
>
> 当前客户端已通过降低 VAD 分段阈值（默认 50s）来规避此限制。

### 3.2 客户端使用

```bash
# 推荐：在项目根目录使用 .env 管理
cp .env.example .env
# 填写 OPENAI_API_KEY / OPENAI_BASE_URL / QWEN3_ASR_MODEL

# 最简用法（自动读取 .env）
qwen3-asr -i /path/to/audio.mp3

# 完整参数
qwen3-asr -i /path/to/audio.mp3 \
    -key <your_api_key> \
    -url http://172.28.245.150:10010/v1 \
    -m Qwen3-ASR-1.7B \
    -j 8 \
    -d 50 \
    -srt

# 或通过环境变量（与 .env 等价）
export OPENAI_API_KEY="<your_api_key>"
export OPENAI_BASE_URL="http://172.28.245.150:10010/v1"
qwen3-asr -i /path/to/audio.mp3
```

### 3.3 Python 直接调用

```python
from qwen3_asr_toolkit.qwen3asr import QwenASR

asr = QwenASR(
    model="Qwen3-ASR-1.7B",
    base_url="http://172.28.245.150:10010/v1",
    api_key="<your_api_key>",
)

language, text = asr.asr("/path/to/audio.wav")
print(f"Language: {language}")
print(f"Text: {text}")
```

---

## 4. 测试验证

以 `sample/` 目录下的三个测试文件验证：

| 文件 | 时长 | VAD 分段 | 语言检测 | 转写结果 |
|---|---|---|---|---|
| `sample_0.mp3` | 9s | 1（无需分段） | Chinese ✅ | ✅ 正确 |
| `deutsch.mp3` | 206s (3.4min) | 4 段 | German ✅ | ✅ 正确 |
| `sample_2.m4a` | 1516s (25min) | 29 段 | Chinese ✅ | ✅ 正确（~24KB 文本）|

---

## 5. 已知限制与注意事项

1. **encoder cache 限制**：vLLM 默认 encoder cache 为 2048 tokens，单段音频不能超过约 150 秒。客户端已通过 VAD 分段（默认 50s）规避，但如果调大 `-d` 参数超过此限制会导致 400 错误。

2. **`asr_options` 不可用**：DashScope 的 `enable_lid`（语言检测）和 `enable_itn`（逆文本正则化）选项在 vLLM OpenAI 兼容 API 中不可配置。vLLM 的 Qwen3-ASR 会自动输出语言标识（嵌入在文本中），ITN 默认不启用。

3. **音频编码开销**：本地文件通过 base64 编码传输，会增大约 33% 的数据量。对于大文件（>10M）会先转 mp3 压缩后再编码。

4. **vLLM 官方参考文档**：[https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html)

---

## 6. 回退方案

如需回退到 DashScope 版本：

```bash
git checkout HEAD~1 -- qwen3_asr_toolkit/qwen3asr.py qwen3_asr_toolkit/call_api.py requirements.txt setup.py
pip install dashscope
pip uninstall openai
```
