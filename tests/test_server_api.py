import io
import wave
import unittest

import numpy as np
from fastapi.testclient import TestClient

from qwen3_asr_toolkit.server import create_app


def _tiny_wav_bytes():
    samples = (np.zeros(1600, dtype=np.int16)).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples)
    return buffer.getvalue()


class DummyOfflineTranscriber:
    def transcribe_file(self, input_file, context="", use_forced_aligner=False):
        return {
            "language": "Chinese",
            "text": "hello",
            "segment_count": 1,
            "audio_duration_sec": 0.1,
            "segments": [
                {
                    "index": 0,
                    "start_sec": 0.0,
                    "end_sec": 0.1,
                    "duration_sec": 0.1,
                    "language": "Chinese",
                    "text": "hello",
                }
            ],
            "forced_aligner": {
                "requested": use_forced_aligner,
                "available": False,
                "message": "not deployed",
                "language": "Chinese",
                "items": [],
                "text": "hello",
            },
        }


class DummyStreamingTranscriber:
    def configure_session(self, session, **kwargs):
        session.context = kwargs.get("context", "")
        return {
            "event": "started",
            "session": {
                "context": session.context,
                "decode_interval_ms": kwargs.get("decode_interval_ms", 600),
                "finalize_silence_ms": kwargs.get("finalize_silence_ms", 600),
            },
        }

    def push_audio(self, session, pcm16k):
        session.language = "Chinese"
        session.text = "partial"
        return {
            "event": "partial",
            "updated": True,
            "language": "Chinese",
            "text": "partial",
            "committed_text": "",
            "live_text": "partial",
            "segment_index": 0,
            "audio_duration_sec": 0.5,
        }

    def finish(self, session):
        session.closed = True
        return {
            "event": "final",
            "language": "Chinese",
            "text": "final",
            "committed_text": "final",
            "live_text": "",
            "audio_duration_sec": 1.0,
        }


class ServerApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.state.offline_transcriber = DummyOfflineTranscriber()
        self.app.state.streaming_transcriber = DummyStreamingTranscriber()
        self.client = TestClient(self.app)

    def test_offline_rest_api(self):
        files = {"audio_file": ("tiny.wav", _tiny_wav_bytes(), "audio/wav")}
        data = {"context": "test", "use_forced_aligner": "true"}
        response = self.client.post("/api/v1/offline/transcribe", files=files, data=data)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["language"], "Chinese")
        self.assertEqual(payload["segment_count"], 1)
        self.assertTrue(payload["forced_aligner"]["requested"])

    def test_realtime_websocket_api(self):
        with self.client.websocket_connect("/ws/v1/realtime/transcribe") as ws:
            ready = ws.receive_json()
            self.assertEqual(ready["event"], "ready")
            self.assertIn("protocol_version", ready)
            self.assertIn("realtime_alignment_supported", ready)

            ws.send_json({"event": "start", "context": "meeting"})
            started = ws.receive_json()
            self.assertEqual(started["event"], "started")
            self.assertIn("session", started)

            ws.send_bytes(np.zeros(1600, dtype=np.float32).tobytes())
            partial = ws.receive_json()
            self.assertEqual(partial["event"], "partial")
            self.assertTrue(partial["updated"])
            self.assertIn("committed_text", partial)
            self.assertIn("live_text", partial)

            ws.send_json({"event": "finish"})
            final = ws.receive_json()
            self.assertEqual(final["event"], "final")
            self.assertEqual(final["text"], "final")
            self.assertIn("committed_text", final)
            self.assertIn("live_text", final)

    def test_realtime_websocket_rejects_forced_aligner_field(self):
        with self.client.websocket_connect("/ws/v1/realtime/transcribe") as ws:
            ws.receive_json()
            ws.send_json({"event": "start", "use_forced_aligner": True})
            error = ws.receive_json()
            self.assertEqual(error["event"], "error")
            self.assertIn("forced aligner", error["message"].lower())


if __name__ == "__main__":
    unittest.main()
