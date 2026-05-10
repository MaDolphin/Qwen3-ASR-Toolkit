import argparse
import unittest
from unittest.mock import Mock, patch

import numpy as np

from qwen3_asr_toolkit import cli, offline_cli, realtime_cli


class CliClientTests(unittest.TestCase):
    def test_offline_cli_default_url(self):
        args = offline_cli.parse_args(["--input-file", "sample/sample_0.mp3"])
        self.assertEqual(args.api_url, "http://127.0.0.1:10012/api/v1/offline/transcribe")
        self.assertEqual(args.timeout_sec, 1800.0)

    def test_realtime_cli_default_url(self):
        args = realtime_cli.parse_args(["--input-file", "sample/sample_2.m4a"])
        self.assertEqual(args.ws_url, "ws://127.0.0.1:10012/ws/stream")
        self.assertEqual(args.chunk_ms, 500)
        self.assertEqual(args.max_inflight_chunks, 4)

    def test_realtime_start_payload_uses_native_protocol(self):
        args = argparse.Namespace(
            context="ctx",
            chunk_size_sec=1.0,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
        )
        payload = realtime_cli.build_start_payload(args)
        self.assertEqual(payload["event"], "start")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["context"], "ctx")
        self.assertNotIn("decode_interval_ms", payload)

    def test_offline_cli_posts_file_and_returns_json(self):
        response = Mock()
        response.json.return_value = {
            "language": "Chinese",
            "text": "你好",
            "segment_count": 1,
            "audio_duration_sec": 1.0,
            "forced_aligner": {"requested": False, "available": False, "items": []},
        }
        response.raise_for_status.return_value = None
        args = offline_cli.parse_args(["--input-file", "sample/sample_0.mp3", "--quiet"])
        with patch("qwen3_asr_toolkit.offline_cli.requests.post", return_value=response) as post:
            result = offline_cli.run(args)
        self.assertEqual(result["text"], "你好")
        self.assertEqual(post.call_args.kwargs["data"]["use_forced_aligner"], "false")

    def test_health_uses_native_base_url(self):
        response = Mock()
        response.json.return_value = {"status": "ok"}
        response.raise_for_status.return_value = None
        with patch("qwen3_asr_toolkit.cli.requests.get", return_value=response) as get:
            cli.run_health(argparse.Namespace(base_url="http://127.0.0.1:10012", timeout_sec=3))
        self.assertEqual(get.call_args.args[0], "http://127.0.0.1:10012/health")

    def test_size_parser(self):
        from deploy.vllm_streaming_server_native import _parse_size_bytes

        self.assertEqual(_parse_size_bytes("8GiB"), 8 * 1024**3)
        self.assertEqual(_parse_size_bytes("8192MiB"), 8 * 1024**3)
        self.assertIsNone(_parse_size_bytes(""))


if __name__ == "__main__":
    unittest.main()
