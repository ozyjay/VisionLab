from __future__ import annotations

import json
import unittest

from src.detectors.object_detector import Detection
from src.scene_state import build_scene_state


class SceneStateTests(unittest.TestCase):
    def test_scene_state_counts_and_privacy_metadata(self) -> None:
        state = build_scene_state(
            frame_id=42,
            fps=29.876,
            object_detections=[
                Detection("person", 0.91, (1, 2, 30, 40), "object"),
                Detection("person", 0.82, (5, 6, 35, 45), "object"),
                Detection("cup", 0.77, (10, 20, 50, 80), "object"),
            ],
            face_detections=[
                Detection("face", 0.99, (3, 4, 20, 25), "face"),
            ],
            face_detection_enabled=True,
            privacy_blur_enabled=True,
            timestamp="2026-07-09T03:00:00Z",
        )

        self.assertEqual(state.timestamp, "2026-07-09T03:00:00Z")
        self.assertEqual(state.frame_id, 42)
        self.assertEqual(state.fps, 29.88)
        self.assertEqual(state.people_count, 2)
        self.assertEqual(state.face_count, 1)
        self.assertEqual(state.object_counts, {"cup": 1, "person": 2})
        self.assertEqual(len(state.detections), 4)
        self.assertEqual(state.privacy["face_detection_enabled"], True)
        self.assertEqual(state.privacy["privacy_blur_enabled"], True)
        self.assertEqual(state.privacy["identity_recognition_enabled"], False)
        self.assertEqual(state.privacy["face_embeddings_stored"], False)

    def test_scene_state_serialises_compact_jsonl(self) -> None:
        state = build_scene_state(
            frame_id=1,
            fps=12.0,
            object_detections=[
                Detection("person", 0.91234, (1, 2, 3, 4), "object"),
            ],
            face_detections=[],
            face_detection_enabled=False,
            privacy_blur_enabled=False,
            timestamp="2026-07-09T03:00:00Z",
        )

        line = state.to_jsonl()
        parsed = json.loads(line)

        self.assertNotIn("\n", line)
        self.assertNotIn(": ", line)
        self.assertEqual(parsed["people_count"], 1)
        self.assertEqual(parsed["detections"][0]["confidence"], 0.9123)
        self.assertEqual(parsed["detections"][0]["bbox"], [1, 2, 3, 4])

    def test_scene_state_ignores_non_matching_detection_sources_for_counts(self) -> None:
        state = build_scene_state(
            frame_id=1,
            fps=1.0,
            object_detections=[
                Detection("person", 1.0, (0, 0, 1, 1), "face"),
            ],
            face_detections=[
                Detection("face", 1.0, (0, 0, 1, 1), "object"),
            ],
            face_detection_enabled=False,
            privacy_blur_enabled=False,
            timestamp="2026-07-09T03:00:00Z",
        )

        self.assertEqual(state.people_count, 0)
        self.assertEqual(state.face_count, 0)
        self.assertEqual(state.object_counts, {})
        self.assertEqual(state.detections, [])


if __name__ == "__main__":
    unittest.main()
