from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from client.cli.url_utils import build_offline_api_url, build_stream_ws_url
from client.gradio.audio_stream import ensure_float32_pcm_16k
from client.gradio.offline_client import OfflineTranscribeError, transcribe_offline
from client.gradio.realtime_client import RealtimeClientError, RealtimeWSClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-ASR Gradio 客户端。")
    parser.add_argument("--server", default="http://127.0.0.1:10012", help="Native ASR 服务地址。")
    parser.add_argument("--api-url", default="", help="离线 HTTP API 地址；优先级高于 --server。")
    parser.add_argument("--ws-url", default="", help="WebSocket 地址；优先级高于 --server。")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio 监听地址。")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 监听端口。")
    parser.add_argument("--share", action="store_true", help="启用 Gradio share。")
    parser.add_argument("--chunk-ms", type=int, default=500, help="实时音频 chunk 毫秒数。")
    parser.add_argument("--chunk-size-sec", type=float, default=1.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--receive-timeout-sec", type=float, default=300.0)
    return parser


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


def create_app(
    server: str = "http://127.0.0.1:10012",
    api_url: str = "",
    ws_url: str = "",
    chunk_size_sec: float = 1.0,
    unfixed_chunk_num: int = 2,
    unfixed_token_num: int = 5,
    receive_timeout_sec: float = 300.0,
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

    def start_realtime(context: str):
        client = RealtimeWSClient(
            resolved_ws_url,
            context=context or "",
            chunk_size_sec=chunk_size_sec,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            receive_timeout_sec=receive_timeout_sec,
        )
        try:
            started = client.start()
            return client, "", "", {"status": "started", "started": started, "ws_url": resolved_ws_url}
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
            realtime_state = gr.State(value=None)
            realtime_context = gr.Textbox(label="上下文 Context", value="")
            with gr.Row():
                start_button = gr.Button("开始实时转写")
                stop_button = gr.Button("停止实时转写")
            mic_audio = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="浏览器麦克风")
            partial_text = gr.Textbox(label="实时 Partial", lines=8)
            final_text = gr.Textbox(label="Final", lines=8)
            status_json = gr.JSON(label="连接状态/指标")
            start_button.click(
                start_realtime,
                inputs=[realtime_context],
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
    try:
        demo = create_app(
            server=args.server,
            api_url=args.api_url,
            ws_url=args.ws_url,
            chunk_size_sec=args.chunk_size_sec,
            unfixed_chunk_num=args.unfixed_chunk_num,
            unfixed_token_num=args.unfixed_token_num,
            receive_timeout_sec=args.receive_timeout_sec,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
