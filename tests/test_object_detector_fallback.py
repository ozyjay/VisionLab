from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from src.detectors.object_detector import ObjectDetector


class ObjectDetectorFallbackTests(unittest.TestCase):
    def test_missing_model_is_safe_fallback(self) -> None:
        detector = ObjectDetector(model_path="models/definitely_missing_model.pt")

        self.assertFalse(detector.available)
        self.assertIn("not found", detector.status_message)
        self.assertEqual(detector.detect(None), [])

    def test_disabled_backend_is_safe_fallback(self) -> None:
        detector = ObjectDetector(model_path=None, backend="none")

        self.assertFalse(detector.available)
        self.assertEqual(detector.detect(None), [])

    def test_yoloe_backend_sets_prompt_classes(self) -> None:
        seen: dict[str, object] = {}

        class FakeModel:
            names: dict[int, str] = {}

            def __init__(self, model_path: str) -> None:
                seen["model_path"] = model_path

            def set_classes(self, prompts: list[str]) -> None:
                seen["prompts"] = prompts
                self.names = {index: prompt for index, prompt in enumerate(prompts)}

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLOE = FakeModel

        fake_accelerator = types.SimpleNamespace(
            resolved_device="cpu",
            backend="cpu",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "yoloe-26s-seg.pt"
            model_path.write_bytes(b"fake")

            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                with patch(
                    "src.detectors.object_detector.get_torch_accelerator_status",
                    return_value=fake_accelerator,
                ):
                    detector = ObjectDetector(
                        model_path=str(model_path),
                        backend="yoloe",
                        prompts=["phone", " keys ", ""],
                    )

        self.assertTrue(detector.available)
        self.assertEqual(seen["model_path"], str(model_path))
        self.assertEqual(seen["prompts"], ["phone", "keys"])
        self.assertIn("yoloe", detector.status_message)
        self.assertIn("prompts=2", detector.status_message)


if __name__ == "__main__":
    unittest.main()
