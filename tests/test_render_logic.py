from __future__ import annotations

import unittest

import numpy as np

from src.config import AppConfig
from src.detectors.object_detector import Detection
from src.main import _draw_overlay, _should_draw_face_boxes, _viewer_window_flags, cv2


class RenderLogicTests(unittest.TestCase):
    def test_face_boxes_are_drawn_when_face_detection_is_enabled(self) -> None:
        self.assertTrue(_should_draw_face_boxes(face_detection_enabled=True))

    def test_face_boxes_are_hidden_when_face_detection_is_disabled(self) -> None:
        self.assertFalse(_should_draw_face_boxes(face_detection_enabled=False))

    @unittest.skipIf(cv2 is None, "OpenCV is not installed")
    def test_viewer_window_uses_plain_resizable_gui_when_supported(self) -> None:
        flags = _viewer_window_flags()

        if cv2.WINDOW_NORMAL != 0:
            self.assertTrue(flags & cv2.WINDOW_NORMAL)
        if hasattr(cv2, "WINDOW_GUI_NORMAL"):
            self.assertTrue(flags & cv2.WINDOW_GUI_NORMAL)

    @unittest.skipIf(cv2 is None, "OpenCV is not installed")
    def test_dashboard_overlay_returns_larger_layout_canvas(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        display = _draw_overlay(
            frame,
            fps=30.0,
            frame_id=7,
            config=AppConfig(),
            face_detection_enabled=True,
            face_detections=[Detection("face", 1.0, (10, 10, 30, 30), "face")],
            face_status="Face detector ready",
            object_detection_enabled=True,
            object_detections=[Detection("person", 0.9, (20, 20, 60, 90), "object")],
            object_status="Object detector ready",
            privacy_blur=False,
            show_help=True,
            last_capture=None,
        )

        self.assertGreater(display.shape[0], frame.shape[0])
        self.assertGreater(display.shape[1], frame.shape[1])
        self.assertEqual(display.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
