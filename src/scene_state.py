"""Scene-state model placeholder for future MVP stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass(slots=True)
class SceneState:
    """Structured scene state.

    MVP 4 will populate this from object and face detections.
    """

    timestamp: str
    frame_id: int
    fps: float
    people_count: int = 0
    face_count: int = 0
    object_counts: dict[str, int] = field(default_factory=dict)
    detections: list[dict[str, Any]] = field(default_factory=list)
    privacy: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialise scene state to JSON."""

        return json.dumps(asdict(self), indent=2, sort_keys=True)
