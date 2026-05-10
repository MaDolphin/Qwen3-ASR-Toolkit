# coding=utf-8
"""内嵌 qwen_asr 包的提示入口。

生产服务请使用项目根命令 `qwen3-asr-native-server`。
"""


def main() -> None:
    print(
        "qwen_asr 是 Qwen3-ASR-Toolkit 内嵌的模型实现包。\n"
        "生产服务入口：qwen3-asr-native-server\n"
        "源码入口：deploy/vllm_streaming_server_native.py"
    )


if __name__ == "__main__":
    main()
