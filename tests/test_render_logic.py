from __future__ import annotations

import unittest

from src.main import _should_draw_face_boxes


class RenderLogicTests(unittest.TestCase):
    def test_face_boxes_are_drawn_when_face_detection_is_enabled(self) -> None:
        self.assertTrue(_should_draw_face_boxes(face_detection_enabled=True))

    def test_face_boxes_are_hidden_when_face_detection_is_disabled(self) -> None:
        self.assertFalse(_should_draw_face_boxes(face_detection_enabled=False))


if __name__ == "__main__":
    unittest.main()
