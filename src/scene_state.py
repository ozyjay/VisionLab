"""Structured scene-state summaries for the local vision demo."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from typing import Any

from .detectors.object_detector import Detection


@dataclass(slots=True)
class SceneState:
    """Structured scene state that excludes image and biometric data."""

    timestamp: str
    frame_id: int
    fps: float
    people_count: int = 0
    face_count: int = 0
    object_counts: dict[str, int] = field(default_factory=dict)
    detections: list[dict[str, Any]] = field(default_factory=list)
    privacy: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialise scene state to pretty JSON for debugging."""

        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def to_jsonl(self) -> str:
        """Serialise scene state to one compact JSONL-compatible line."""

        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)


def _utc_timestamp() -> str:
    """Return a compact UTC timestamp for scene-state snapshots."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _detection_to_dict(detection: Detection) -> dict[str, Any]:
    """Convert a detection to JSON-safe scene-state data."""

    return {
        "label": detection.label,
        "confidence": round(float(detection.confidence), 4),
        "bbox": [int(value) for value in detection.bbox],
        "source": detection.source,
    }


def build_scene_state(
    *,
    frame_id: int,
    fps: float,
    object_detections: list[Detection],
    face_detections: list[Detection],
    face_detection_enabled: bool,
    privacy_blur_enabled: bool,
    timestamp: str | None = None,
) -> SceneState:
    """Build a scene-state snapshot from current non-identifying detections."""

    objects = [detection for detection in object_detections if detection.source == "object"]
    faces = [detection for detection in face_detections if detection.source == "face"]
    object_counts = Counter(detection.label for detection in objects)

    return SceneState(
        timestamp=timestamp or _utc_timestamp(),
        frame_id=int(frame_id),
        fps=round(float(fps), 2),
        people_count=object_counts.get("person", 0),
        face_count=len(faces),
        object_counts=dict(sorted(object_counts.items())),
        detections=[_detection_to_dict(detection) for detection in [*objects, *faces]],
        privacy={
            "face_detection_enabled": bool(face_detection_enabled),
            "privacy_blur_enabled": bool(privacy_blur_enabled),
            "identity_recognition_enabled": False,
            "face_embeddings_stored": False,
        },
    )
