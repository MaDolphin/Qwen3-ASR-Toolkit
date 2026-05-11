import unittest
from unittest.mock import Mock, patch

import numpy as np

from client.gradio.audio_stream import ensure_float32_pcm_16k
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


if __name__ == "__main__":
    unittest.main()
