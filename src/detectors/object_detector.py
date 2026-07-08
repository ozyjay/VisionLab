"""Object detector interface and no-model fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Detection:
    """Single detection result."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    source: str


class ObjectDetector:
    """MVP fallback object detector.

    Real YOLO/ONNX inference is planned for MVP 2. This class keeps the public
    interface stable while returning no detections when no model is configured.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.available = False

    def detect(self, frame: Any) -> list[Detection]:
        """Return object detections for a frame."""

        _ = frame
        return []
