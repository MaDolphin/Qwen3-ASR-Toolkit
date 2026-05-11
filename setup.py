from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "numpy",
    "librosa",
    "soundfile",
    "pydub",
    "silero_vad",
    "tqdm",
    "requests",
    "srt",
    "fastapi",
    "uvicorn",
    "python-multipart",
    "websockets",
    "python-dotenv",
    "transformers==4.57.6",
    "nagisa==0.2.11",
    "soynlp==0.0.493",
    "accelerate==1.12.0",
    "qwen-omni-utils",
    "sox",
    "pytz",
    "scipy",
    "vllm==0.14.0",
    "modelscope",
    "setuptools<81.0.0,>=77.0.3",
]

setup(
    name="qwen3-asr-toolkit",
    version="2.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "qwen_asr": ["inference/assets/*.dict"],
    },
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "native": INSTALL_REQUIRES,
        "client": [
            "gradio",
            "librosa",
            "numpy",
            "soundfile",
            "websockets",
            "requests",
        ],
    },
    entry_points={
        "console_scripts": [
            "qwen3-asr-native-server=deploy.vllm_streaming_server_native:main",
            "qwen3-asr-offline-cli=client.cli.offline:main",
            "qwen3-asr-stream-cli=client.cli.stream:main",
            "qwen3-asr-cli=client.cli.main:main",
            "qwen3-asr-gradio=client.gradio.app:main",
        ],
        "vllm.general_plugins": [
            "qwen3_asr=qwen_asr.vllm_plugin:register_qwen3_asr_model",
        ],
    },
    author="He Wang",
    author_email="hwang2001@mail.nwpu.edu.cn",
    description="Unified native Qwen3-ASR server for offline HTTP and realtime WebSocket transcription.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/QwenLM/Qwen3-ASR-Toolkit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
