import unittest
from unittest.mock import patch

import numpy as np

from qwen3_asr_toolkit.offline_transcriber import OfflineTranscriber, merge_languages


class DummyASRClient:
    def __init__(self):
        self.calls = []

    def asr_waveform(self, wav, sr=16000, context=""):
        self.calls.append((len(wav), sr, context))
        return "Chinese", f"text_{len(self.calls)}"


class OfflineTranscriberTests(unittest.TestCase):
    def test_merge_languages_deduplicates_adjacent_values(self):
        self.assertEqual(merge_languages(["Chinese", "Chinese", "English", "", "English"]), "Chinese,English")

    def test_short_audio_uses_single_segment(self):
        client = DummyASRClient()
        transcriber = OfflineTranscriber(asr_client=client, num_threads=1)
        wav = np.zeros(16000 * 3, dtype=np.float32)

        result = transcriber.transcribe_waveform(wav, context="meeting", use_forced_aligner=False)

        self.assertEqual(result["segment_count"], 1)
        self.assertEqual(result["language"], "Chinese")
        self.assertEqual(result["text"], "text_1")
        self.assertEqual(client.calls[0][2], "meeting")

    def test_enforces_max_segment_duration_after_vad(self):
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

    def test_falls_back_to_fixed_split_when_vad_unavailable(self):
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

    def test_forced_aligner_disabled_metadata(self):
        client = DummyASRClient()
        transcriber = OfflineTranscriber(asr_client=client, aligner_base_url=None)
        wav = np.zeros(16000 * 3, dtype=np.float32)

        result = transcriber.transcribe_waveform(wav, use_forced_aligner=True)
        aligner = result["forced_aligner"]

        self.assertTrue(aligner["requested"])
        self.assertFalse(aligner["available"])
        self.assertIn("not deployed", aligner["message"].lower())


if __name__ == "__main__":
    unittest.main()
