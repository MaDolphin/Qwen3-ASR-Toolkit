# 开发者指南

本文档面向后续维护和扩展 `Qwen3-ASR-Toolkit` 的开发者，重点说明当前主路径、代码结构、关键模块职责、请求链路和开发建议。

## 1. 先理解主路径

当前仓库里最重要的是两套代码：

- `qwen3_asr_toolkit/`
  - 这是当前项目主实现
  - 服务端、CLI、离线/实时转写、远程 forced aligner 客户端都在这里
- `qwen_asr/`
  - 这是从上游迁入的官方包
  - 主要用于保留官方能力和示例兼容

如果你要改“当前系统对外提供的功能”，优先看：

- [qwen3_asr_toolkit/server.py](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/qwen3_asr_toolkit/server.py)
- [qwen3_asr_toolkit/offline_transcriber.py](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/qwen3_asr_toolkit/offline_transcriber.py)
- [qwen3_asr_toolkit/streaming_transcriber.py](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/qwen3_asr_toolkit/streaming_transcriber.py)

## 2. 目录职责

### 2.1 `qwen3_asr_toolkit/`

当前业务主实现：

- `server.py`
  - FastAPI 服务入口
  - 提供 REST + WebSocket
- `qwen3asr.py`
  - ASR 主模型调用封装
  - 调用 Qwen3-ASR vLLM OpenAI 兼容接口
- `offline_transcriber.py`
  - 离线长音频转写总控
  - 负责 VAD、并发调用、结果聚合、可选对齐
- `streaming_transcriber.py`
  - 实时会话状态机
  - 负责累计音频与增量重解码
- `forced_aligner_client.py`
  - 远程 forced aligner 客户端
  - 调用 `/tokenize` + `/pooling`
- `audio_tools.py`
  - 音频加载、VAD、存储辅助函数
- `offline_cli.py`
  - REST 离线客户端
- `realtime_cli.py`
  - WS 实时客户端
- `call_api.py`
  - 本地直连 vLLM 的离线 CLI
- `env_utils.py`
  - `.env` 加载与配置辅助

### 2.2 `qwen_asr/`

上游包嵌入：

- 保留上游能力与结构
- 不建议把当前业务逻辑继续堆在这里
- 更适合作为参考实现或兼容入口

### 2.3 `examples/`

- `java-example/`
  - Java 版本示例
- `upstream-qwen3-asr/`
  - 上游示例快照
  - 不是当前主系统的业务入口

### 2.4 `tests/`

当前测试覆盖：

- 离线切分逻辑
- VAD 回退
- 流式状态机
- REST/WS 协议
- `qwen_asr` 嵌入后的导入行为

## 3. 请求链路

### 3.1 离线链路

链路如下：

1. `server.py` 接收 `POST /api/v1/offline/transcribe`
2. 写入临时文件
3. `OfflineTranscriber.transcribe_file()`
4. `load_audio()`
5. 长音频走 `process_vad()`
6. 每段并发调用 `QwenASR.asr_waveform()`
7. 聚合文本
8. 若 `use_forced_aligner=true`，调用 `RemoteForcedAlignerClient`
9. 返回最终 JSON

### 3.2 实时链路

链路如下：

1. WebSocket 建立连接
2. 创建 `StreamingSession`
3. 客户端持续上传 `float32` PCM chunk
4. `StreamingTranscriber.push_audio()`
5. 每达到一个解码步长，就把累计音频交给 `OfflineTranscriber.transcribe_waveform(..., use_forced_aligner=False)`
6. 返回 partial
7. `finish` 时返回 final

注意：

- 实时模式固定不做对齐
- 这是产品边界，不建议偷偷在实时链路里插入 forced aligner

## 4. 配置来源

当前配置统一来自项目根 `.env`。

关键变量：

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `QWEN3_ASR_MODEL`
- `QWEN3_ALIGNER_BASE_URL`
- `QWEN3_ALIGNER_API_KEY`
- `QWEN3_ALIGNER_MODEL`

配置加载入口：

- [env_utils.py](/Users/hhk/Github/Self/Qwen3-ASR-Toolkit/qwen3_asr_toolkit/env_utils.py)

## 5. 当前设计约束

### 5.1 长音频硬约束

离线模式保证：

- 每段最终长度 `<= 60s`

即使：

- VAD 返回异常分段
- VAD 初始化失败

也会通过回退切分保证这个约束成立。

### 5.2 对齐边界

只有离线支持对齐：

- REST 离线接口
- 离线 CLI
- 本地直连 CLI

实时模式无对齐：

- WebSocket API
- 实时 CLI

### 5.3 Forced Aligner 的调用方式

当前远程 aligner 不走 `chat/completions`。

而是：

- `/tokenize`
- `/pooling`

并且使用：

- `task=token_classify`

这是当前实现里非常重要的事实，后续不要误改成 `chat/completions`。

## 6. 开发建议

### 6.1 改接口前先看文档和测试

建议顺序：

1. 看 `doc/USAGE.md`
2. 看已有测试
3. 再改服务代码

### 6.2 新功能优先放在 `qwen3_asr_toolkit/`

除非你明确在维护上游兼容逻辑，否则不要优先改 `qwen_asr/`。

### 6.3 先守住边界再扩功能

最重要的边界有三个：

- 长音频切分上限 `60s`
- 对齐只在离线生效
- 实时链路输入必须是 `float32 16k mono`

## 7. 常用开发命令

运行测试：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

语法检查：

```bash
python -m compileall qwen3_asr_toolkit qwen_asr tests
```

本地启动服务：

```bash
python -m qwen3_asr_toolkit.server --host 127.0.0.1 --port 18000
```

运行样例验证：

```bash
bash scripts/run_sample_tests.sh
```

## 8. 推荐后续工作

如果你准备继续开发，优先级建议：

1. 优化 forced aligner 时间戳后处理
2. 增加 systemd / Docker 一键部署能力
3. 增加更完整的集成测试
4. 增加请求耗时与错误分类日志
