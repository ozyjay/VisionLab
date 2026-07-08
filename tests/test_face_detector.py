from __future__ import annotations

import unittest
from unittest.mock import patch

from src.config import AppConfig
from src.detectors.face_detector import FaceDetector


class FaceDetectorTests(unittest.TestCase):
    def test_no_opencv_is_safe_fallback(self) -> None:
        with patch("src.detectors.face_detector.cv2", None):
            detector = FaceDetector()

        self.assertFalse(detector.available)
        self.assertIn("OpenCV", detector.status_message)
        self.assertEqual(detector.detect(None), [])

    def test_missing_explicit_model_is_safe_fallback(self) -> None:
        detector = FaceDetector(model_path="models/definitely_missing_face_model.onnx")

        self.assertIsInstance(detector.available, bool)
        self.assertEqual(detector.detect(None), [])

    def test_active_modes_describes_faces_as_off_not_placeholder(self) -> None:
        config = AppConfig(enable_face_detection=False)

        self.assertIn("faces:off", config.active_modes)
        self.assertNotIn("placeholder", config.active_modes)

    def test_default_face_model_path_uses_local_cascade(self) -> None:
        config = AppConfig.from_env()

        self.assertEqual(
            config.face_model_path,
            "models/face_detection_yunet_2026may.onnx",
        )


if __name__ == "__main__":
    unittest.main()
