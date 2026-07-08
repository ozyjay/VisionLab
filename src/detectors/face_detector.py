"""Face detector interface and no-model fallback."""

from __future__ import annotations

from typing import Any

from .object_detector import Detection


class FaceDetector:
    """MVP fallback face detector.

    Real face detection and privacy blur are planned for MVP 3. This class
    preserves the detector interface without implementing identity recognition.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.available = False

    def detect(self, frame: Any) -> list[Detection]:
        """Return face detections for a frame."""

        _ = frame
        return []
