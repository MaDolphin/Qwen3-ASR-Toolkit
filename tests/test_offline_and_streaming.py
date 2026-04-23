import unittest
from unittest.mock import patch

import numpy as np

from qwen3_asr_toolkit.offline_transcriber import OfflineTranscriber
from qwen3_asr_toolkit.streaming_transcriber import StreamingSession, StreamingTranscriber


class DummyASRClient:
    def __init__(self):
        self.calls = []

    def asr_waveform(self, wav, sr=16000, context=""):
        self.calls.append((len(wav), sr, context))
        return "Chinese", f"text_{len(self.calls)}"


class OfflineAndStreamingTests(unittest.TestCase):
    def test_offline_transcribe_enforces_max_60s_per_segment(self):
        client = DummyASRClient()
        transcriber = OfflineTranscriber(
            asr_client=client,
            num_threads=1,
            vad_target_segment_s=45,
            vad_max_segment_s=60,
        )
        wav = np.zeros(16000 * 125, dtype=np.float32)

        bad_segments = [
            (0, 16000 * 80, wav[: 16000 * 80]),
            (16000 * 80, 16000 * 125, wav[16000 * 80 :]),
        ]
        with patch("qwen3_asr_toolkit.offline_transcriber.load_silero_vad", return_value=object()), patch(
            "qwen3_asr_toolkit.offline_transcriber.process_vad", return_value=bad_segments
        ):
            result = transcriber.transcribe_waveform(wav, context="", use_forced_aligner=False)

        self.assertEqual(result["segment_count"], 3)
        self.assertTrue(all(seg["duration_sec"] <= 60.0 for seg in result["segments"]))

    def test_offline_transcribe_forced_aligner_unavailable(self):
        client = DummyASRClient()
        transcriber = OfflineTranscriber(asr_client=client, aligner_base_url=None)
        wav = np.zeros(16000 * 3, dtype=np.float32)

        result = transcriber.transcribe_waveform(wav, use_forced_aligner=True)
        aligner = result["forced_aligner"]

        self.assertTrue(aligner["requested"])
        self.assertFalse(aligner["available"])
        self.assertIn("not deployed", aligner["message"].lower())

    def test_offline_transcribe_fallback_when_vad_unavailable(self):
        client = DummyASRClient()
        transcriber = OfflineTranscriber(
            asr_client=client,
            num_threads=1,
            vad_target_segment_s=45,
            vad_max_segment_s=60,
        )
        wav = np.zeros(16000 * 121, dtype=np.float32)

        with patch("qwen3_asr_toolkit.offline_transcriber.load_silero_vad", side_effect=RuntimeError("vad init failed")):
            result = transcriber.transcribe_waveform(wav)

        self.assertEqual(result["segment_count"], 3)
        self.assertTrue(all(seg["duration_sec"] <= 60.0 for seg in result["segments"]))

    def test_streaming_session_updates_and_finalizes(self):
        client = DummyASRClient()
        offline = OfflineTranscriber(asr_client=client, num_threads=1)
        streaming = StreamingTranscriber(offline_transcriber=offline, decode_interval_ms=200, finalize_silence_ms=600)
        session = StreamingSession(context="meeting")

        step = np.ones(16000, dtype=np.float32) * 0.02
        partial = streaming.push_audio(session, step)
        self.assertEqual(partial["event"], "partial")
        self.assertTrue(partial["updated"])
        self.assertEqual(partial["language"], "Chinese")

        final = streaming.finish(session)
        self.assertEqual(final["event"], "final")
        self.assertEqual(final["language"], "Chinese")
        self.assertGreaterEqual(final["audio_duration_sec"], 1.0)

    def test_streaming_session_emits_segment_final_after_silence(self):
        client = DummyASRClient()
        offline = OfflineTranscriber(asr_client=client, num_threads=1)
        streaming = StreamingTranscriber(offline_transcriber=offline, decode_interval_ms=200, finalize_silence_ms=400)
        session = StreamingSession(context="meeting")

        speech = np.ones(16000, dtype=np.float32) * 0.02
        silence = np.zeros(8000, dtype=np.float32)

        partial = streaming.push_audio(session, speech)
        self.assertEqual(partial["event"], "partial")
        self.assertTrue(partial["updated"])

        event = streaming.push_audio(session, silence)
        self.assertEqual(event["event"], "segment_final")
        self.assertIn("segment_text", event)
        self.assertEqual(event["live_text"], "")
        self.assertEqual(event["committed_text"], event["text"])

    def test_streaming_finish_returns_committed_and_live_text(self):
        client = DummyASRClient()
        offline = OfflineTranscriber(asr_client=client, num_threads=1)
        streaming = StreamingTranscriber(offline_transcriber=offline, decode_interval_ms=200, finalize_silence_ms=600)
        session = StreamingSession(context="meeting")

        speech = np.ones(6400, dtype=np.float32) * 0.02
        streaming.push_audio(session, speech)
        final = streaming.finish(session)

        self.assertEqual(final["event"], "final")
        self.assertIn("committed_text", final)
        self.assertIn("live_text", final)
        self.assertEqual(final["committed_text"], final["text"])


if __name__ == "__main__":
    unittest.main()
