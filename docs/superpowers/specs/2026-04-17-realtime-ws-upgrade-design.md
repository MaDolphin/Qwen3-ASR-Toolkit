# 实时 WebSocket 准实时升级设计

## 目标

在保持当前远端 vLLM `chat.completions` 部署方式不变的前提下，升级项目的实时 WebSocket 转写能力，使其更接近流式体验：

- 升级 WebSocket 协议
- 引入接收队列与后台处理循环
- 避免“整段历史音频反复重转写”
- 仅离线支持 forced aligner，实时模式继续不支持对齐

## 现状

当前实现的实时模式通过 `WebSocket -> 累计整段音频 -> 周期性调用 OfflineTranscriber.transcribe_waveform()` 工作。

问题在于：

- 随会话变长，重复转写的历史音频越来越多
- 延迟和成本会逐步升高
- 协议表达能力较弱，不便于扩展状态与错误语义
- 接收和推理未完全解耦，后续提升空间有限

## 约束

- 后端 ASR 推理为远端 vLLM OpenAI-compatible 接口
- 当前服务未暴露官方 `/v1/realtime` 端点
- 因此无法直接复用 `qwen_asr.Qwen3ASRModel` 的本地 streaming state
- 必须在项目内实现“准实时”处理策略

## 方案

### 1. 协议升级

保留路径 `/ws/v1/realtime/transcribe`，升级事件语义：

- 服务端发送 `ready`
- 客户端发送 `start`
- 客户端持续发送二进制音频帧
- 服务端发送 `partial`
- 服务端在检测到分段完成时发送 `segment_final`
- 客户端发送 `finish`
- 服务端发送 `final`

`ready` 返回能力信息：

- `session_id`
- `sample_rate`
- `audio_format`
- `protocol_version`
- `realtime_alignment_supported=false`
- 默认实时参数

`start` 支持参数：

- `context`
- `decode_interval_ms`
- `min_chunk_ms`
- `finalize_silence_ms`
- `max_segment_sec`

### 2. 服务端处理模型

每个会话采用三层状态：

- `incoming_queue`：WebSocket 收到的原始音频块
- `segment_audio`：当前正在识别的未完成语音段
- `committed_text`：已经最终确认的文本

服务端拆成两个协程：

- `receiver`：专门负责从 WebSocket 接收文本帧和音频帧，并写入队列
- `worker`：专门负责消费队列、合并音频、触发转写和回传结果

这样可以避免推理期间阻塞接收。

### 3. 准实时策略

由于不能使用模型原生 streaming state，本次采用“当前语音段增量重解码 + 静音提交”的策略：

- 音频小包持续进入 `segment_audio`
- 每到 `decode_interval_ms` 对当前段做一次转写，返回 `partial`
- 当检测到尾部静音达到 `finalize_silence_ms` 时，将当前段作为一个最终段提交
- 提交后文本并入 `committed_text`，然后清空当前段
- 若当前段持续过长，则在 `max_segment_sec` 达到时强制提交

### 4. 静音检测

使用轻量级 RMS 能量阈值来做实时尾静音检测：

- 输入为最新收到的音频块
- 统计连续静音时长
- 超过阈值后触发 `segment_final`

这样实现简单、代价低，也不依赖远端模型能力。

### 5. 文本输出模型

会话文本由两部分组成：

- `committed_text`：已最终确认
- `live_text`：当前未完成段的最新转写

对外返回：

- `partial.text = committed_text + live_text`
- `segment_final.text = committed_text`
- `final.text = committed_text + live_text`

### 6. 实时模式边界

- 不支持 forced aligner
- 若客户端在 WS `start` 中传入对齐相关字段，服务端返回显式错误
- 长音频离线的 VAD 分段逻辑保留在离线接口
- 实时模式不再调用离线长音频整段 VAD 切分

## 测试策略

覆盖以下行为：

- 新协议 `ready/start/partial/segment_final/final`
- 接收与处理解耦后的基本链路
- 静音提交逻辑
- 强制分段逻辑
- 实时模式拒绝对齐参数
- CLI 能正确适配新协议

## 风险

- RMS 静音检测比模型原生 streaming state 更粗糙
- 过低阈值可能误切，过高阈值可能延迟提交
- 需要通过配置项暴露关键参数，便于线上调优

## 结论

本次不追求“官方 realtime 端点级别的真流式”，而是先在现有远端部署条件下，把项目升级为更稳妥、更低重复计算、更易扩展的准实时转写服务。
