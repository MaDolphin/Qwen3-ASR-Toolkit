# Streaming Server 改进记录

> **文档用途**：记录本次对两个 Streaming Server（HTTP 后端版 / Native 本地模型版）的修改过程、技术决策和验证结果，供项目移交和维护参考。
>
> **修改日期**：2026-04-23
> **涉及文件**：
> - `deploy/streaming_utils.py`（新增）
> - `deploy/vllm_streaming_server.py`
> - `deploy/vllm_streaming_server_native.py`

---

## 一、背景与问题

项目原有两种实时转写服务端：

| 服务端 | 模式 | 模型加载方式 |
|--------|------|-------------|
| `deploy/vllm_streaming_server.py` | HTTP 后端 | 不加载模型，调用远端 vLLM `/v1/audio/transcriptions?stream=true` |
| `deploy/vllm_streaming_server_native.py` | Native 本地 | 通过 `Qwen3ASRModel.LLM()` 在进程内加载 vLLM 模型 |

在真实数据验证中发现以下问题：

1. **HTTP 后端长音频输出包含重复前缀**：`deutsch.mp3`（206s）通过 `/v1/audio/transcriptions?stream=true` 流式返回时，SSE token 流中会多次出现 `language German<asr_text>` 前缀，导致最终文本被污染。
2. **Native 服务端配置偏保守**：`max_new_tokens=32` 对于较大 chunk 或快速语速容易截断输出。
3. **两个服务端协议不一致**：HTTP 服务端在 `started` 事件中返回 `sample_rate` 和 `audio_format`，并在每个音频 chunk 后返回 `ack`；Native 服务端缺少这些字段，不利于客户端统一对接。
4. **代码重复**：音频缓冲、float32 校验、事件构造等逻辑在两个服务端中重复实现。

---

## 二、修改内容

### 2.1 新增共享工具模块 `deploy/streaming_utils.py`

提取两个服务端的公共逻辑，避免重复实现：

```python
def validate_audio_chunk(raw_bytes: bytes) -> Tuple[bool, Optional[str], Optional[np.ndarray]]
def accumulate_buffer(buffer: np.ndarray, chunk: np.ndarray) -> np.ndarray
def consume_full_chunks(buffer: np.ndarray, chunk_size_samples: int) -> Tuple[np.ndarray, list]
def clean_duplicate_asr_prefixes(text: str) -> str
def build_ack_event(received_samples: int, total_samples: int) -> dict
def build_started_event(...) -> dict
```

**核心函数说明**：

- `clean_duplicate_asr_prefixes(text)`  
  用正则 `language\s+\S+\s*<asr_text>` 匹配所有语言前缀，保留第一个，删除后续重复。  
  该函数专门用于解决 vLLM SSE 流式端点在长音频场景下间歇性重新输出语言标记的问题。

- `consume_full_chunks(buffer, chunk_size_samples)`  
  从缓冲区中切出尽可能多的完整 chunk，返回 `(剩余缓冲区, 已消费 chunk 列表)`。  
  简化 Native 服务端的手动 while 循环。

- `build_started_event()` / `build_ack_event()`  
  统一构造 WebSocket 控制事件，确保两个服务端返回的 JSON 结构一致。

### 2.2 修改 HTTP Streaming Server

**文件**：`deploy/vllm_streaming_server.py`

**变更点**：

1. **导入共享工具模块**，替换原有内联逻辑。
2. **新增前缀清理逻辑**：在 SSE token 流累积 `full_raw` 的每一步后调用 `clean_duplicate_asr_prefixes()`：

```python
async for token, finish_reason in _parse_sse_stream(resp):
    full_raw += token
    full_raw = clean_duplicate_asr_prefixes(full_raw)  # 新增
    await websocket.send_json({
        "event": "token",
        "token": token,
        "text_so_far": full_raw,
    })
```

3. **统一事件格式**：`started` 事件和 `ack` 事件改为调用 `build_started_event()` / `build_ack_event()`。

### 2.3 修改 Native Streaming Server

**文件**：`deploy/vllm_streaming_server_native.py`

**变更点**：

1. **`max_new_tokens` 默认值从 `32` 提升到 `128`**：
   - 原值 32 在 `chunk_size_sec=1.0` 且语速较快时容易截断输出。
   - 提升后覆盖更大 chunk 和更快语速场景，同时不会显著增加显存占用（因为每次只生成新增部分）。

2. **导入共享工具模块**，替换原有内联逻辑。

3. **`started` 事件补全字段**：新增 `sample_rate` 和 `audio_format`，与 HTTP 服务端保持一致：

```python
await websocket.send_json(
    build_started_event(
        stream=True,
        chunk_size_sec=chunk_size_sec,
        unfixed_chunk_num=unfixed_chunk_num,
        unfixed_token_num=unfixed_token_num,
    )
)
```

4. **每个音频 chunk 后发送 `ack` 事件**：客户端现在可以确认服务器已收到音频数据。

5. **简化 chunk 消费逻辑**：使用 `consume_full_chunks()` 替代手动 while 循环。

---

## 三、关键技术决策

### 3.1 为什么用客户端正则清理，而不是修复 vLLM 服务端？

vLLM `/v1/audio/transcriptions?stream=true` 的 SSE token 流在长音频（>60s）场景下会间歇性地重新输出 `language {Lang}<asr_text>` 前缀。这是 vLLM 生成引擎在 token-level streaming 时的行为，属于模型/引擎层面的现象，短期内无法通过服务端配置消除。

因此选择在客户端（HTTP Streaming Server）做后处理，用正则去重。该处理是**幂等**的：即使 vLLM 未来修复了该问题，清理函数也不会破坏正常输出。

### 3.2 为什么 Native 服务端必须用 `Qwen3ASRModel.LLM()`，不能用 `vllm serve`？

这是项目移交中最容易混淆的一点，必须明确：

| 维度 | `Qwen3ASRModel.LLM()` | `vllm serve` |
|------|----------------------|--------------|
| API 类型 | Python 进程内 API (`vllm.LLM.generate()`) | HTTP REST API |
| 模型位置 | 与 Streaming Server **同一进程** | 独立进程 |
| KV Cache | **有状态** — 引擎在多次 `generate()` 调用间复用 prefix KV Cache | **无状态** — 每个 HTTP 请求独立计算 |
| 延迟特性 | 每 chunk 延迟 ≈ O(新增音频长度) | 每 chunk 延迟 ≈ O(累积音频长度) |

`streaming_transcribe()` 的核心机制是**增量重喂 + KV Cache 复用**：

1. 每收到一个音频 chunk，服务端把「从开始到现在的全部音频」重新喂给模型。
2. 但 prompt 中包含之前已解码的文本作为 prefix。
3. vLLM 引擎检测到 prefix 与上次调用相同，自动复用 prefix 的 KV Cache，只计算新增部分的 KV。
4. 因此每个 chunk 的推理时间几乎恒定，不会随音频长度增长。

如果改用 `vllm serve` 的 HTTP 端点：
- 每个 chunk 都是独立的 HTTP POST，vLLM 每次都要从头计算整个音频的 KV Cache。
- 延迟会随音频长度线性增长，长音频下完全无法满足实时性要求。
- vLLM 当前没有提供「跨请求复用 KV Cache」的 HTTP API。

**结论**：要实现真正的低延迟实时 streaming，Native Server 必须在进程内通过 `Qwen3ASRModel.LLM()` 加载模型。**`vllm serve` 和 Native Streaming 在架构上是互斥的。**

### 3.3 部署模式选择

| 目标 | 推荐部署方式 | 说明 |
|------|-------------|------|
| **真实时 streaming**（低延迟） | 单独运行 `vllm_streaming_server_native.py` | 它在进程内加载模型，不需要也不应该同时运行 `vllm serve`（会抢 GPU 显存导致 OOM） |
| **OpenAI 兼容 HTTP API**（batch/offline） | 单独运行 `vllm serve` | 提供标准 `/v1/audio/transcriptions` 接口 |
| **两者都要** | 需要 **两块 GPU**（或单卡显存足够大） | GPU-0 跑 `vllm serve`，GPU-1 跑 Native Streaming Server |

---

## 四、验证结果

### 4.1 单元测试

全部 12 个现有单元测试通过，无回归：

```
tests/test_offline_and_streaming.py     6 passed
tests/test_qwen_asr_embedding.py        3 passed
tests/test_server_api.py                3 passed
```

### 4.2 HTTP Streaming 长音频验证

使用 `deutsch.mp3`（206.2s）通过 HTTP Streaming Server 测试：

| 指标 | 结果 |
|------|------|
| Final `text` 中重复前缀数量 | **0** ✅ |
| Final `raw` 中前缀出现次数 | **1**（仅第一个合法前缀）✅ |
| Streaming 过程中 dirty token 事件 | **0** ✅ |
| Token 总数 | 871 |
| Finish→first-token 延迟 | ~8.9s（符合预期，HTTP 模式需等音频收完） |

### 4.3 共享工具模块自测

`clean_duplicate_asr_prefixes()` 覆盖以下场景：

| 输入 | 输出 |
|------|------|
| `language German<asr_text>Hello` | `language German<asr_text>Hello`（无重复，不变） |
| `language German<asr_text>Hello language German<asr_text>World` | `language German<asr_text>Hello World` |
| `language German<asr_text>A language English<asr_text>B` | `language German<asr_text>A B`（后续不同语言前缀也清除） |
| `No prefix at all` | `No prefix at all`（无匹配，不变） |
| 三重重复前缀 | 仅保留第一个 |

---

## 五、文件变更清单

```
deploy/
├── streaming_utils.py                      [新增] 共享工具模块
├── vllm_streaming_server.py                [修改] 增加前缀清理 + 使用共享工具
└── vllm_streaming_server_native.py         [修改] max_new_tokens=128 + 协议一致性 + 使用共享工具
```

---

## 六、后续维护建议

1. **若 vLLM 未来修复了 SSE 流式前缀重复问题**：`clean_duplicate_asr_prefixes()` 仍然是安全的（幂等），可以保留作为防御性代码。
2. **若需要调整 chunk 大小**：Native Server 的 `max_new_tokens` 可能需要同步调整。建议按 `chunk_size_sec * 15` 估算（假设每秒约 10-15 个 token）。
3. **新增 streaming server 时**：优先复用 `deploy/streaming_utils.py`，保持协议一致性。
4. **显存规划**：Native Server 启动时会独占 GPU 显存（由 `gpu_memory_utilization` 控制，默认 0.8）。如果同一台机器还需要跑其他服务，务必预留足够显存，或降低该参数。
