from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest.mock import patch

from src.config import AppConfig


class ConfigTests(unittest.TestCase):
    def test_camera_resolution_mode_defaults_to_fast(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = AppConfig.from_env()

        self.assertEqual(config.camera_resolution_mode, "fast")
        self.assertEqual(config.camera_resolution, (640, 480))
        self.assertIn("camera:fast", config.active_modes)

    def test_camera_resolution_mode_can_select_quality(self) -> None:
        with patch.dict(os.environ, {"VISION_CAMERA_RESOLUTION_MODE": "quality"}):
            config = AppConfig.from_env()

        self.assertEqual(config.camera_resolution_mode, "quality")
        self.assertEqual(config.camera_resolution, (1280, 720))
        self.assertIn("camera:quality", config.active_modes)

    def test_invalid_camera_resolution_mode_falls_back_to_fast(self) -> None:
        output = io.StringIO()
        with patch.dict(os.environ, {"VISION_CAMERA_RESOLUTION_MODE": "cinema"}):
            with contextlib.redirect_stdout(output):
                config = AppConfig.from_env()

        self.assertEqual(config.camera_resolution_mode, "fast")
        self.assertEqual(config.camera_resolution, (640, 480))
        self.assertIn("unsupported", output.getvalue())

    def test_object_prompts_parse_comma_separated_values(self) -> None:
        env = {"VISION_OBJECT_PROMPTS": " phone, keys ,wallet,, remote control "}
        with patch.dict(os.environ, env):
            config = AppConfig.from_env()

        self.assertEqual(
            config.object_prompts,
            ["phone", "keys", "wallet", "remote control"],
        )

    def test_object_detection_hold_frames_can_be_configured(self) -> None:
        with patch.dict(os.environ, {"VISION_OBJECT_DETECTION_HOLD_FRAMES": "12"}):
            config = AppConfig.from_env()

        self.assertEqual(config.object_detection_hold_frames, 12)

    def test_face_detection_interval_can_be_configured(self) -> None:
        with patch.dict(os.environ, {"VISION_FACE_DETECTION_INTERVAL": "4"}):
            config = AppConfig.from_env()

        self.assertEqual(config.face_detection_interval, 4)

    def test_web_dashboard_port_can_be_configured(self) -> None:
        with patch.dict(os.environ, {"VISION_WEB_PORT": "8020"}):
            config = AppConfig.from_env()

        self.assertEqual(config.web_port, 8020)


if __name__ == "__main__":
    unittest.main()
