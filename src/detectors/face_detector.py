"""Non-identifying face detector interface and OpenCV backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..camera import cv2
from .object_detector import Detection


class FaceDetector:
    """Local face detector that never performs identity recognition.

    The detector uses OpenCV YuNet when a local ONNX model is available, with a
    Haar cascade fallback for OpenCV builds that still include it. It returns
    generic ``face`` boxes only; it does not compute embeddings, compare faces,
    name people, or persist biometric data.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.available = False
        self.status_message = "Face detection is unavailable."
        self._backend = "none"
        self._detector: Any | None = None

        if cv2 is None:
            self.status_message = "OpenCV is not installed; face detection is unavailable."
            return

        model = self._resolve_model_path(model_path)
        if model is None:
            self.status_message = (
                "No local OpenCV face detector model was found; face detection is unavailable."
            )
            return

        if model.suffix.lower() == ".onnx" and hasattr(cv2, "FaceDetectorYN_create"):
            try:
                self._detector = cv2.FaceDetectorYN_create(
                    str(model),
                    "",
                    (320, 320),
                    0.8,
                    0.3,
                    5000,
                )
            except Exception as exc:
                self.status_message = f"OpenCV YuNet face detector failed to load: {exc}"
                return

            self._backend = "yunet"
            self.available = True
            self.status_message = (
                f"Face detector ready: OpenCV YuNet model={model}; "
                "identity recognition and embeddings are disabled"
            )
            return

        if model.suffix.lower() == ".xml" and hasattr(cv2, "CascadeClassifier"):
            classifier = cv2.CascadeClassifier(str(model))
            if classifier.empty():
                self.status_message = f"OpenCV face cascade failed to load: {model}"
                return

            self._detector = classifier
            self._backend = "cascade"
            self.available = True
            self.status_message = (
                f"Face detector ready: OpenCV Haar cascade={model.name}; "
                "identity recognition and embeddings are disabled"
            )
            return

        self.status_message = (
            f"Unsupported face detector model or OpenCV build for {model}. "
            "Use a YuNet ONNX model with OpenCV FaceDetectorYN support."
        )

    def _resolve_model_path(self, model_path: str | None) -> Path | None:
        """Return a usable local face detector model path."""

        if model_path:
            explicit_path = Path(model_path)
            if explicit_path.exists():
                return explicit_path

        default_yunet_path = Path("models/face_detection_yunet_2026may.onnx")
        if default_yunet_path.exists():
            return default_yunet_path

        data_dir = getattr(cv2, "data", None)
        haar_dir = getattr(data_dir, "haarcascades", None)
        if haar_dir:
            bundled_path = Path(haar_dir) / "haarcascade_frontalface_default.xml"
            if bundled_path.exists():
                return bundled_path

        return None

    def detect(self, frame: Any) -> list[Detection]:
        """Return face detections for a frame."""

        if not self.available or self._detector is None or frame is None:
            return []

        try:
            if self._backend == "yunet":
                height, width = frame.shape[:2]
                self._detector.setInputSize((int(width), int(height)))
                _, faces = self._detector.detect(frame)
                return self._detections_from_yunet(faces)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self._detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            return self._detections_from_cascade(faces)
        except Exception as exc:
            self.available = False
            self.status_message = f"Face detector failed; disabling detector: {exc}"
            return []

    def _detections_from_yunet(self, faces: Any) -> list[Detection]:
        """Convert YuNet rows to generic face detections."""

        if faces is None:
            return []

        detections: list[Detection] = []
        for face in faces:
            x, y, width, height = (float(value) for value in face[:4])
            confidence = float(face[-1])
            if confidence < 0.8:
                continue
            detections.append(
                Detection(
                    label="face",
                    confidence=confidence,
                    bbox=(
                        int(round(x)),
                        int(round(y)),
                        int(round(x + width)),
                        int(round(y + height)),
                    ),
                    source="face",
                )
            )
        return detections

    def _detections_from_cascade(self, faces: Any) -> list[Detection]:
        """Convert Haar cascade rectangles to generic face detections."""

        detections: list[Detection] = []
        for x, y, width, height in faces:
            detections.append(
                Detection(
                    label="face",
                    confidence=1.0,
                    bbox=(int(x), int(y), int(x + width), int(y + height)),
                    source="face",
                )
            )

        return detections
