from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.camera import Camera


class CameraTests(unittest.TestCase):
    def test_set_resolution_updates_open_capture_and_reports_actual_size(self) -> None:
        capture = Mock()
        capture.get.side_effect = [1920.0, 1080.0]
        camera = Camera(resolution=(640, 480))
        camera.capture = capture
        fake_cv2 = Mock(CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4)

        with patch("src.camera.cv2", fake_cv2):
            actual = camera.set_resolution((1920, 1080))

        self.assertEqual(camera.resolution, (1920, 1080))
        self.assertEqual(actual, (1920, 1080))
        capture.set.assert_any_call(3, 1920.0)
        capture.set.assert_any_call(4, 1080.0)


if __name__ == "__main__":
    unittest.main()
