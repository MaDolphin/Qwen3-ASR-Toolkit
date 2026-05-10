import unittest

import numpy as np

import deploy.vllm_streaming_server_native as native_server


class FakeResult:
    language = "Chinese"
    text = "你好"


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, context, return_time_stamps=False):
        self.calls.append(
            {
                "audio": audio,
                "context": context,
                "return_time_stamps": return_time_stamps,
            }
        )
        return [FakeResult()]


class DummySemaphore:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class NativeServerApiTests(unittest.TestCase):
    def test_import_does_not_load_model(self):
        self.assertIsNone(native_server._qwen_asr_model)

    def test_routes_are_registered(self):
        paths = {route.path for route in native_server.app.routes}
        self.assertIn("/health", paths)
        self.assertIn("/api/v1/offline/transcribe", paths)
        self.assertIn("/ws/stream", paths)

    def test_capabilities_shape(self):
        capabilities = native_server._capabilities()
        self.assertIn("offline_http", capabilities)
        self.assertIn("native_websocket", capabilities)
        self.assertIn("asr_model_loaded_once", capabilities)
        self.assertTrue(capabilities["native_websocket"])

    def test_native_adapter_requires_16k(self):
        adapter = native_server.NativeQwenASRAdapter(lambda: FakeModel(), DummySemaphore())
        with self.assertRaises(ValueError):
            adapter.asr_waveform(np.zeros(1600, dtype=np.float32), sr=8000)

    def test_native_adapter_calls_shared_model(self):
        model = FakeModel()
        adapter = native_server.NativeQwenASRAdapter(lambda: model, DummySemaphore())

        language, text = adapter.asr_waveform(np.zeros(1600, dtype=np.float32), sr=16000, context="ctx")

        self.assertEqual(language, "Chinese")
        self.assertEqual(text, "你好")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["context"], ["ctx"])
        self.assertFalse(model.calls[0]["return_time_stamps"])


if __name__ == "__main__":
    unittest.main()
