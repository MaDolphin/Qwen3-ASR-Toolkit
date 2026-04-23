# Java 示例（DashScope 旧版兼容）

> ⚠️ **重要提示**：本 Java 示例基于原始 **DashScope** API（`qwen3-asr-flash`）实现，使用 `dashscope-sdk-java` SDK。
> 
> 当前主项目（v2.0.0+）已迁移至 **vLLM OpenAI 兼容接口**，Python 主实现不再依赖 DashScope。如需将 Java 客户端接入新版服务端，建议参考以下 REST/WebSocket 接口自行实现：
> - 离线 REST API：`POST /api/v1/offline/transcribe`
> - 实时 WebSocket API：`ws://<host>:<port>/ws/v1/realtime/transcribe`
> 
> 接口协议细节请参阅项目根目录的 [`doc/cloud_api_ws_usage.md`](../../doc/cloud_api_ws_usage.md)。

## 功能说明

本示例包含两部分能力：

1. **Silero VAD ONNX 推理**（`App.java`）
   - 纯 Java 实现，不依赖 Python
   - 使用 ONNX Runtime 加载 `silero_vad.onnx`
   - 与 Python 版 `get_speech_timestamps` 逻辑保持一致
   - 支持 16kHz WAV 文件语音活动检测

2. **DashScope ASR 长音频转写**（`Test.java`）
   - 调用阿里云 DashScope `qwen3-asr-flash` 模型
   - 支持本地文件与远程 URL 输入
   - 长音频自动按 VAD 切分后多线程并行调用
   - 结果聚合并保存为 `.txt`

## 运行环境

- Java 11+
- Maven 3.6+
- FFmpeg（用于音频格式转换与时长获取）

## 构建

```bash
cd examples/java-example
mvn clean package
```

## 运行 VAD 示例

修改 `App.java` 中的以下路径后执行：

```java
private static final String MODEL_PATH = "path/to/silero_vad.onnx";
private static final String AUDIO_FILE_PATH = "path/to/audio.wav";
```

```bash
mvn exec:java -Dexec.mainClass="org.example.App"
```

## 运行 ASR 示例

修改 `Test.java` 中的以下参数后执行：

```java
String url = "path/to/audio.mp3";           // 本地路径或 HTTP(S) URL
String modelName = "qwen3-asr-flash";       // DashScope 模型名
String context = "";                        // 可选上下文提示
String apiKey = "";                         // DashScope API Key（留空则取环境变量）
int numThreads = 4;                         // 并行线程数
int vadDuration = 120;                      // VAD 目标分段时长（秒）
```

```bash
mvn exec:java -Dexec.mainClass="org.example.Test"
```

## 目录结构

```
src/main/java/org/example/
├── App.java                 # Silero VAD ONNX 推理入口
├── Test.java                # DashScope ASR 长音频转写入口
├── AudioSplitter.java       # 音频切分与 FFmpeg 调用工具
├── SlieroVadDetector.java   # VAD 检测器封装
├── SlieroVadOnnxModel.java  # ONNX 模型加载与推理
└── utils/
    ├── Util.java            # ASR API 调用、音频转换、文件操作
    └── Result.java          # 通用结果封装
```

## 注意事项

- `silero_vad.onnx` 模型文件已放置在 `src/main/resources/` 下，但 `App.java` 中硬编码的 `MODEL_PATH` 需要按你本地实际路径修改。
- `Test.java` 中的临时文件默认写入当前工作目录，Windows 下使用 `\` 作为路径分隔符。
- 如需接入新版 vLLM 服务，核心改动点是将 `util.QWen3Asr()` 内的 DashScope SDK 调用替换为 `OkHttp` + Jackson 对 OpenAI 兼容 API 的 HTTP 请求。
