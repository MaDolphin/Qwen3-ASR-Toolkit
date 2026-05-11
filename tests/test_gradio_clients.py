import unittest
from unittest.mock import Mock, patch

import numpy as np

from client.gradio.audio_stream import ensure_float32_pcm_16k
from client.gradio.app import HTTP_MIC_ACCESS_STEPS, _language_default, _launch_kwargs, _realtime_language_config, build_parser
from client.gradio.offline_client import transcribe_offline
from client.gradio.realtime_client import RealtimeWSClient


class GradioClientTests(unittest.TestCase):
    def test_audio_stream_int16_stereo_to_float32_16k(self):
        audio = np.array([[0, 1000], [1000, 0], [-1000, -1000]], dtype=np.int16)
        wav = ensure_float32_pcm_16k((16000, audio))
        self.assertEqual(wav.dtype, np.float32)
        self.assertEqual(wav.ndim, 1)
        self.assertEqual(len(wav), 3)
        self.assertLessEqual(float(np.max(np.abs(wav))), 1.0)

    def test_audio_stream_resamples_to_16k(self):
        wav = ensure_float32_pcm_16k((8000, np.ones(8000, dtype=np.float32)))
        self.assertEqual(wav.dtype, np.float32)
        self.assertGreaterEqual(len(wav), 15900)
        self.assertLessEqual(len(wav), 16100)

    def test_offline_client_posts_multipart(self):
        response = Mock()
        response.json.return_value = {"text": "你好"}
        response.raise_for_status.return_value = None
        with patch("client.gradio.offline_client.requests.post", return_value=response) as post:
            result = transcribe_offline("http://server/api", "sample/sample_0.mp3", context="ctx", use_forced_aligner=True)
        self.assertEqual(result["text"], "你好")
        self.assertEqual(post.call_args.kwargs["data"]["context"], "ctx")
        self.assertEqual(post.call_args.kwargs["data"]["use_forced_aligner"], "true")

    def test_realtime_start_payload(self):
        client = RealtimeWSClient("ws://127.0.0.1:10012/ws/stream", context="ctx")
        payload = client.build_start_payload()
        self.assertEqual(payload["event"], "start")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["context"], "ctx")
        self.assertEqual(payload["chunk_size_sec"], 1.0)

    def test_realtime_start_payload_includes_forced_language(self):
        client = RealtimeWSClient("ws://127.0.0.1:10012/ws/stream", context="ctx", language="Chinese")
        payload = client.build_start_payload()
        self.assertEqual(payload["language"], "Chinese")

    def test_realtime_client_disables_auto_proxy_by_default(self):
        client = RealtimeWSClient("ws://127.0.0.1:10012/ws/stream")
        kwargs = client._connect_kwargs()
        self.assertIsNone(kwargs.get("proxy"))
        self.assertIsNone(kwargs["max_size"])
        self.assertIsNone(kwargs["ping_interval"])

    def test_gradio_parser_defaults_to_remote_bind(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 7860)
        self.assertEqual(args.realtime_language_1, "Chinese")
        self.assertEqual(args.realtime_language_2, "English")

    def test_gradio_parser_accepts_explicit_remote_endpoints(self):
        args = build_parser().parse_args([
            "--api-url",
            "http://172.28.245.150:10010/api/v1/offline/transcribe",
            "--ws-url",
            "ws://172.28.245.150:10010/ws/stream",
        ])
        self.assertEqual(args.api_url, "http://172.28.245.150:10010/api/v1/offline/transcribe")
        self.assertEqual(args.ws_url, "ws://172.28.245.150:10010/ws/stream")

    def test_gradio_launch_kwargs_include_https_options(self):
        args = build_parser().parse_args([
            "--host",
            "0.0.0.0",
            "--port",
            "7860",
            "--ssl-keyfile",
            "/etc/qwen3-asr/gradio.key",
            "--ssl-certfile",
            "/etc/qwen3-asr/gradio.crt",
            "--ssl-keyfile-password",
            "secret",
            "--ssl-no-verify",
        ])
        kwargs = _launch_kwargs(args)
        self.assertEqual(kwargs["server_name"], "0.0.0.0")
        self.assertEqual(kwargs["server_port"], 7860)
        self.assertEqual(kwargs["ssl_keyfile"], "/etc/qwen3-asr/gradio.key")
        self.assertEqual(kwargs["ssl_certfile"], "/etc/qwen3-asr/gradio.crt")
        self.assertEqual(kwargs["ssl_keyfile_password"], "secret")
        self.assertFalse(kwargs["ssl_verify"])

    def test_realtime_language_config_for_single_language(self):
        context, forced_language, candidates = _realtime_language_config("ctx", "Chinese", "自动识别")
        self.assertEqual(context, "ctx")
        self.assertEqual(forced_language, "Chinese")
        self.assertEqual(candidates, ["Chinese"])

    def test_realtime_language_config_for_two_languages(self):
        context, forced_language, candidates = _realtime_language_config("会议术语", "Chinese", "English")
        self.assertEqual(forced_language, "")
        self.assertEqual(candidates, ["Chinese", "English"])
        self.assertIn("会议术语", context)
        self.assertIn("Chinese 和 English", context)

    def test_language_default_rejects_unknown_value(self):
        self.assertEqual(_language_default("Chinese"), "Chinese")
        self.assertEqual(_language_default("Unknown"), "自动识别")
        self.assertEqual(_language_default(""), "自动识别")

    def test_http_mic_access_steps_include_chrome_flag(self):
        self.assertIn("chrome://flags/#unsafely-treat-insecure-origin-as-secure", HTTP_MIC_ACCESS_STEPS)
        self.assertIn("http://服务器IP:10012", HTTP_MIC_ACCESS_STEPS)
        self.assertIn("Relaunch", HTTP_MIC_ACCESS_STEPS)


if __name__ == "__main__":
    unittest.main()
