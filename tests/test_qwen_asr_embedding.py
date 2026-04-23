import unittest

import qwen_asr
from qwen_asr import parse_asr_output


class QwenAsrEmbeddingTests(unittest.TestCase):
    def test_parse_asr_output_available(self):
        language, text = parse_asr_output("language Chinese<asr_text>你好")
        self.assertEqual(language, "Chinese")
        self.assertEqual(text, "你好")

    def test_lazy_model_import_error_is_actionable(self):
        try:
            _ = qwen_asr.Qwen3ASRModel
            model_available = True
        except Exception as exc:
            model_available = False
            self.assertIn("Failed to import qwen_asr model classes", str(exc))
        self.assertIn(model_available, [True, False])

    def test_lazy_aligner_import_error_is_actionable(self):
        try:
            _ = qwen_asr.Qwen3ForcedAligner
            aligner_available = True
        except Exception as exc:
            aligner_available = False
            self.assertIn("Failed to import qwen_asr model classes", str(exc))
        self.assertIn(aligner_available, [True, False])


if __name__ == "__main__":
    unittest.main()
