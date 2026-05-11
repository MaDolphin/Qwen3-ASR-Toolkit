from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from client.cli.url_utils import build_offline_api_url, build_stream_ws_url
from client.gradio.audio_stream import ensure_float32_pcm_16k
from client.gradio.offline_client import OfflineTranscribeError, transcribe_offline
from client.gradio.realtime_client import RealtimeClientError, RealtimeWSClient


HTTP_MIC_ACCESS_STEPS = """
## 内网 HTTP 访问麦克风设置

如果你通过 `http://服务器IP:端口` 访问本页面且浏览器无法录音，请在 Chrome 中按下面步骤开启临时信任：

1. 打开 `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
2. 在输入框中填写当前 Gradio 地址，例如 `http://服务器IP:10012`
3. 将该实验项设置为 `Enabled`
4. 点击 Chrome 右下角的 `Relaunch` 重启浏览器
5. 重新打开 `http://服务器IP:10012`，允许麦克风权限后再开始实时转写

这是内网调试方案。正式多人使用时，建议改用 HTTPS 或可信内网网关。
""".strip()

AUTO_LANGUAGE = "自动识别"
SUPPORTED_LANGUAGES = [
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-ASR Gradio 客户端。")
    parser.add_argument("--server", default="http://127.0.0.1:10012", help="Native ASR 服务地址。")
    parser.add_argument("--api-url", default="", help="离线 HTTP API 地址；优先级高于 --server。")
    parser.add_argument("--ws-url", default="", help="WebSocket 地址；优先级高于 --server。")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio 监听地址。")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 监听端口。")
    parser.add_argument("--share", action="store_true", help="启用 Gradio share。")
    parser.add_argument("--ssl-keyfile", default="", help="HTTPS 私钥文件路径；远程 IP 使用麦克风时建议配置。")
    parser.add_argument("--ssl-certfile", default="", help="HTTPS 证书文件路径；远程 IP 使用麦克风时建议配置。")
    parser.add_argument("--ssl-keyfile-password", default="", help="HTTPS 私钥密码。")
    parser.add_argument("--ssl-no-verify", action="store_true", help="禁用 Gradio SSL 证书校验，适合内网自签证书调试。")
    parser.add_argument("--chunk-ms", type=int, default=500, help="实时音频 chunk 毫秒数。")
    parser.add_argument("--chunk-size-sec", type=float, default=1.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--receive-timeout-sec", type=float, default=300.0)
    parser.add_argument("--realtime-language-1", default="Chinese", help="实时会议语言预设 1；留空或设为 自动识别 表示自动识别。")
    parser.add_argument("--realtime-language-2", default="English", help="实时会议语言预设 2；最多两种语言。")
    return parser


def _launch_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "ssl_keyfile": args.ssl_keyfile or None,
        "ssl_certfile": args.ssl_certfile or None,
        "ssl_keyfile_password": args.ssl_keyfile_password or None,
        "ssl_verify": not args.ssl_no_verify,
    }


def _segments_table(result: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for segment in result.get("segments") or []:
        rows.append([
            segment.get("index"),
            segment.get("start_sec"),
            segment.get("end_sec"),
            segment.get("duration_sec"),
            segment.get("language"),
            segment.get("text"),
        ])
    return rows


def _selected_languages(language_1: str | None, language_2: str | None) -> list[str]:
    selected: list[str] = []
    for language in (language_1, language_2):
        value = (language or "").strip()
        if value and value != AUTO_LANGUAGE and value not in selected:
            selected.append(value)
    return selected[:2]


def _language_default(language: str | None) -> str:
    value = (language or "").strip()
    if not value or value == AUTO_LANGUAGE:
        return AUTO_LANGUAGE
    return value if value in SUPPORTED_LANGUAGES else AUTO_LANGUAGE


def _realtime_language_config(context: str, language_1: str | None, language_2: str | None) -> tuple[str, str, list[str]]:
    languages = _selected_languages(language_1, language_2)
    base_context = (context or "").strip()
    if len(languages) == 1:
        return base_context, languages[0], languages
    if len(languages) == 2:
        hint = (
            f"会议场景：音频只包含 {languages[0]} 和 {languages[1]} 两种语言。"
            f"请只在这两种语言中识别，不要切换到其他语言。"
        )
        combined_context = f"{base_context}\n{hint}".strip() if base_context else hint
        return combined_context, "", languages
    return base_context, "", []


def create_app(
    server: str = "http://127.0.0.1:10012",
    api_url: str = "",
    ws_url: str = "",
    chunk_size_sec: float = 1.0,
    unfixed_chunk_num: int = 2,
    unfixed_token_num: int = 5,
    receive_timeout_sec: float = 300.0,
    realtime_language_1: str = "Chinese",
    realtime_language_2: str = "English",
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("缺少 Gradio 依赖，请先执行：pip install -r requirements-client.txt") from exc

    resolved_api_url = api_url or build_offline_api_url(server)
    resolved_ws_url = ws_url or build_stream_ws_url(server)

    def run_offline(audio_path: str | None, context: str, use_forced_aligner: bool):
        if not audio_path:
            return "请先上传音频文件。", {"error": "missing audio"}, []
        try:
            result = transcribe_offline(
                resolved_api_url,
                audio_path,
                context=context or "",
                use_forced_aligner=use_forced_aligner,
            )
        except OfflineTranscribeError as exc:
            return f"离线转写失败：{exc}", {"error": str(exc)}, []
        text = result.get("text") or ""
        return text or "转写文本为空。", result, _segments_table(result)

    def start_realtime(context: str, language_1: str, language_2: str):
        realtime_context, forced_language, language_candidates = _realtime_language_config(context, language_1, language_2)
        client = RealtimeWSClient(
            resolved_ws_url,
            context=realtime_context,
            chunk_size_sec=chunk_size_sec,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            receive_timeout_sec=receive_timeout_sec,
            language=forced_language,
        )
        try:
            started = client.start()
            return client, "", "", {
                "status": "started",
                "started": started,
                "ws_url": resolved_ws_url,
                "forced_language": forced_language,
                "language_candidates": language_candidates,
            }
        except RealtimeClientError as exc:
            return None, f"实时连接失败：{exc}", "", {"status": "error", "error": str(exc)}

    def stream_audio(audio, client: RealtimeWSClient | None):
        if client is None:
            return client, "请先点击开始实时转写。", "", {"status": "not_started"}
        if audio is None:
            return client, client.metrics.last_partial, client.metrics.final_text, client.metrics.as_dict()
        try:
            pcm = ensure_float32_pcm_16k(audio)
            metrics = client.send_audio(pcm)
            return client, client.metrics.last_partial, client.metrics.final_text, metrics
        except (RealtimeClientError, ValueError) as exc:
            return client, f"实时转写失败：{exc}", client.metrics.final_text, {"status": "error", "error": str(exc)}

    def stop_realtime(client: RealtimeWSClient | None):
        if client is None:
            return None, "", "", {"status": "not_started"}
        try:
            metrics = client.finish()
            return None, client.metrics.last_partial, client.metrics.final_text, metrics
        except RealtimeClientError as exc:
            client.close()
            return None, client.metrics.last_partial, f"结束失败：{exc}", {"status": "error", "error": str(exc)}

    with gr.Blocks(title="Qwen3-ASR 客户端") as demo:
        gr.Markdown("# Qwen3-ASR 客户端\n\nGradio 仅作为客户端访问已部署的 Native ASR 服务，不加载模型。")
        gr.Markdown(f"- 离线 API：`{resolved_api_url}`\n- 实时 WS：`{resolved_ws_url}`")
        with gr.Tab("离线转写"):
            offline_audio = gr.Audio(type="filepath", label="上传音频文件")
            offline_context = gr.Textbox(label="上下文 Context", value="")
            offline_aligner = gr.Checkbox(label="启用 ForcedAligner 时间戳", value=False)
            offline_button = gr.Button("开始离线转写")
            offline_text = gr.Textbox(label="转写文本", lines=12)
            offline_json = gr.JSON(label="完整响应")
            offline_segments = gr.Dataframe(
                headers=["index", "start_sec", "end_sec", "duration_sec", "language", "text"],
                label="Segments",
            )
            offline_button.click(
                run_offline,
                inputs=[offline_audio, offline_context, offline_aligner],
                outputs=[offline_text, offline_json, offline_segments],
            )
        with gr.Tab("实时转写"):
            gr.Markdown(HTTP_MIC_ACCESS_STEPS)
            realtime_state = gr.State(value=None)
            realtime_context = gr.Textbox(label="上下文 Context", value="")
            language_choices = [AUTO_LANGUAGE, *SUPPORTED_LANGUAGES]
            with gr.Row():
                realtime_language_1 = gr.Dropdown(
                    choices=language_choices,
                    value=_language_default(realtime_language_1),
                    label="会议语言 1",
                )
                realtime_language_2 = gr.Dropdown(
                    choices=language_choices,
                    value=_language_default(realtime_language_2),
                    label="会议语言 2",
                )
            with gr.Row():
                start_button = gr.Button("开始实时转写")
                stop_button = gr.Button("停止实时转写")
            mic_audio = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="浏览器麦克风")
            partial_text = gr.Textbox(label="实时 Partial", lines=8)
            final_text = gr.Textbox(label="Final", lines=8)
            status_json = gr.JSON(label="连接状态/指标")
            start_button.click(
                start_realtime,
                inputs=[realtime_context, realtime_language_1, realtime_language_2],
                outputs=[realtime_state, partial_text, final_text, status_json],
            )
            mic_audio.stream(
                stream_audio,
                inputs=[mic_audio, realtime_state],
                outputs=[realtime_state, partial_text, final_text, status_json],
            )
            stop_button.click(
                stop_realtime,
                inputs=[realtime_state],
                outputs=[realtime_state, partial_text, final_text, status_json],
            )
    return demo


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if bool(args.ssl_keyfile) != bool(args.ssl_certfile):
        print("ERROR: --ssl-keyfile 和 --ssl-certfile 必须同时提供。", file=sys.stderr)
        raise SystemExit(1)
    try:
        demo = create_app(
            server=args.server,
            api_url=args.api_url,
            ws_url=args.ws_url,
            chunk_size_sec=args.chunk_size_sec,
            unfixed_chunk_num=args.unfixed_chunk_num,
            unfixed_token_num=args.unfixed_token_num,
            receive_timeout_sec=args.receive_timeout_sec,
            realtime_language_1=args.realtime_language_1,
            realtime_language_2=args.realtime_language_2,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    demo.launch(**_launch_kwargs(args))


if __name__ == "__main__":
    main()
