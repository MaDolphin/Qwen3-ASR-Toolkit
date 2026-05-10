# 更新日志

## 2.1.0

- 项目收敛为生产精简版，只保留 Unified Native ASR Server。
- `Qwen3-ASR-1.7B` 在同一服务进程中只加载一次，同时提供离线 HTTP 和在线 WebSocket。
- 新增正式命令：`qwen3-asr-native-server`。
- 删除旧 OpenAI API 客户端、旧主服务、旧 vLLM HTTP streaming server、旧 CLI 和上游 examples 快照。
- 文档统一改为中文，并补充模型下载、部署、API、架构和验证说明。
- Forced Aligner 默认禁用，仅在远端模式下按需使用。

## 2.0.0

- 新增 Native WebSocket streaming server。
- 增强长音频 WebSocket 背压和 windowed 验证。
- 增加离线长音频 VAD 分段和 forced aligner metadata。
