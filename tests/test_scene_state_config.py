from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest.mock import patch

from src.config import AppConfig
from src.main import _scene_state_interval_due


class SceneStateConfigTests(unittest.TestCase):
    def test_scene_state_config_defaults_to_on_demand_stdout_only(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VISION_SCENE_STATE_INTERVAL_SECONDS": "",
                "VISION_SCENE_STATE_LOG_PATH": "",
            },
            clear=False,
        ):
            os.environ.pop("VISION_SCENE_STATE_INTERVAL_SECONDS", None)
            os.environ.pop("VISION_SCENE_STATE_LOG_PATH", None)
            config = AppConfig.from_env()

        self.assertEqual(config.scene_state_interval_seconds, 0.0)
        self.assertEqual(config.scene_state_log_path, "")

    def test_scene_state_config_reads_interval_and_log_path(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VISION_SCENE_STATE_INTERVAL_SECONDS": "2.5",
                "VISION_SCENE_STATE_LOG_PATH": "logs/scene-state.jsonl",
            },
        ):
            config = AppConfig.from_env()

        self.assertEqual(config.scene_state_interval_seconds, 2.5)
        self.assertEqual(config.scene_state_log_path, "logs/scene-state.jsonl")

    def test_invalid_scene_state_interval_falls_back_to_disabled(self) -> None:
        output = io.StringIO()
        with patch.dict(os.environ, {"VISION_SCENE_STATE_INTERVAL_SECONDS": "soon"}):
            with contextlib.redirect_stdout(output):
                config = AppConfig.from_env()

        self.assertEqual(config.scene_state_interval_seconds, 0.0)
        self.assertIn("is not a number", output.getvalue())

    def test_negative_scene_state_interval_is_clamped_to_disabled(self) -> None:
        with patch.dict(os.environ, {"VISION_SCENE_STATE_INTERVAL_SECONDS": "-3"}):
            config = AppConfig.from_env()

        self.assertEqual(config.scene_state_interval_seconds, 0.0)

    def test_scene_state_interval_due_logic(self) -> None:
        self.assertFalse(
            _scene_state_interval_due(
                now=10.0,
                last_emit_time=None,
                interval_seconds=0.0,
            )
        )
        self.assertTrue(
            _scene_state_interval_due(
                now=10.0,
                last_emit_time=None,
                interval_seconds=2.0,
            )
        )
        self.assertFalse(
            _scene_state_interval_due(
                now=11.0,
                last_emit_time=10.0,
                interval_seconds=2.0,
            )
        )
        self.assertTrue(
            _scene_state_interval_due(
                now=12.0,
                last_emit_time=10.0,
                interval_seconds=2.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
