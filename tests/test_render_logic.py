from __future__ import annotations

import unittest

import numpy as np

from src.config import AppConfig
from src.detectors.object_detector import Detection
from src.main import (
    _WebDashboardRuntime,
    _backend_for_model_path,
    _build_demo_commentary,
    _clamp_confidence_threshold,
    _dashboard_html,
    _draw_overlay,
    _normalise_prompt_text,
    _should_draw_face_boxes,
    _viewer_window_flags,
    cv2,
)


class RenderLogicTests(unittest.TestCase):
    def test_face_boxes_are_drawn_when_face_detection_is_enabled(self) -> None:
        self.assertTrue(_should_draw_face_boxes(face_detection_enabled=True))

    def test_face_boxes_are_hidden_when_face_detection_is_disabled(self) -> None:
        self.assertFalse(_should_draw_face_boxes(face_detection_enabled=False))

    def test_demo_commentary_mentions_yoloe_prompts(self) -> None:
        config = AppConfig(
            object_detector_backend="yoloe",
            object_model_path="models/yoloe-26s-seg.pt",
            object_prompts=["phone", "keys", "wallet"],
        )

        commentary = _build_demo_commentary(
            config=config,
            face_detection_enabled=False,
            face_detections=[],
            object_detection_enabled=True,
            object_detections=[],
            privacy_blur=False,
            object_status="Object detector ready",
        )

        self.assertTrue(any("YOLOE" in line for line in commentary))
        self.assertTrue(any("phone" in line for line in commentary))

    def test_demo_commentary_distinguishes_blur_from_face_box_display(self) -> None:
        commentary = _build_demo_commentary(
            config=AppConfig(),
            face_detection_enabled=False,
            face_detections=[],
            object_detection_enabled=False,
            object_detections=[],
            privacy_blur=True,
            object_status="Object detector disabled",
        )

        self.assertTrue(any("Face box display is off" in line for line in commentary))
        self.assertFalse(any("Face mode is off unless" in line for line in commentary))

    def test_dashboard_html_uses_browser_rendered_ui(self) -> None:
        html = _dashboard_html()

        self.assertIn("VisionLab dashboard", html)
        self.assertIn("/stream", html)
        self.assertIn("Browser-rendered demo UI", html)
        self.assertIn("modelSelect", html)
        self.assertIn("promptEditor", html)
        self.assertIn("/prompts", html)
        self.assertIn("confidenceSlider", html)
        self.assertIn("/confidence", html)

    def test_dashboard_model_selector_is_above_live_detection_sections(self) -> None:
        html = _dashboard_html()

        self.assertLess(html.index('id="modelSelect"'), html.index('id="objects"'))
        self.assertIn("height: 72px", html)

    def test_prompt_text_is_normalised_from_comma_separated_values(self) -> None:
        self.assertEqual(
            _normalise_prompt_text(" person, pen ,, mobile phone,watch "),
            ["person", "pen", "mobile phone", "watch"],
        )

    def test_confidence_threshold_is_clamped(self) -> None:
        self.assertEqual(_clamp_confidence_threshold(-0.5), 0.0)
        self.assertEqual(_clamp_confidence_threshold(1.5), 1.0)
        self.assertEqual(_clamp_confidence_threshold(0.45), 0.45)

    def test_web_runtime_confidence_threshold_updates_state(self) -> None:
        runtime = _WebDashboardRuntime(AppConfig(object_confidence_threshold=0.35))

        result = runtime.set_confidence_threshold("0.7")

        self.assertTrue(result["ok"])
        self.assertEqual(runtime.config.object_confidence_threshold, 0.7)
        self.assertEqual(runtime.state()["object_confidence_threshold"], 0.7)

    def test_backend_for_model_path_auto_detects_yoloe(self) -> None:
        self.assertEqual(
            _backend_for_model_path("models/yoloe-26s-seg.pt", "auto"),
            "yoloe",
        )
        self.assertEqual(
            _backend_for_model_path("models/yolo11m.pt", "auto"),
            "ultralytics",
        )

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
