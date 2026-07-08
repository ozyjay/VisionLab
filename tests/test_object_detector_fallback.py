from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
