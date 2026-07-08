"""Object detector interface and optional Ultralytics YOLO backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..accelerator import get_torch_accelerator_status


@dataclass(slots=True)
class Detection:
    """Single detection result."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    source: str


class ObjectDetector:
    """Swappable object detector with a safe no-model fallback.

    The current real backend is Ultralytics YOLO. It is loaded only when:

    - object detection is enabled by the caller,
    - the backend is ``ultralytics`` or ``auto``,
    - the model file exists locally, and
    - the optional ``ultralytics`` package imports cleanly.

    Missing model files or missing dependencies leave the detector unavailable
    but do not stop the app.
    """

    def __init__(
        self,
        model_path: str | None = None,
        backend: str = "ultralytics",
        confidence_threshold: float = 0.35,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.backend = backend.strip().lower()
        self.confidence_threshold = confidence_threshold
        self.requested_device = device
        self.accelerator_status = get_torch_accelerator_status(device)
        self.device = self.accelerator_status.resolved_device
        self.available = False
        self.status_message = "Object detection is disabled."
        self._model: Any | None = None
        self._names: dict[int, str] = {}

        if self.backend in {"none", "off", "disabled"}:
            self.status_message = "Object detector backend is disabled."
            return

        if self.backend not in {"auto", "ultralytics"}:
            self.status_message = (
                f"Unsupported object detector backend {self.backend!r}. "
                "Supported backends: ultralytics, auto, none."
            )
            return

        if not model_path:
            self.status_message = (
                "No object detector model path configured. Set VISION_OBJECT_MODEL_PATH."
            )
            return

        path = Path(model_path)
        if not path.exists():
            self.status_message = (
                f"Object detection model not found at {path}. "
                "Detection will stay off until a local YOLO model is available."
            )
            return

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self.status_message = (
                "Ultralytics is not installed. Install optional object detection "
                "dependencies with: python -m pip install -r "
                "requirements-object-detection.txt"
            )
            self._import_error = exc
            return
        except Exception as exc:
            self.status_message = f"Ultralytics import failed: {exc}"
            return

        try:
            self._model = YOLO(str(path))
            names = getattr(self._model, "names", {})
            if isinstance(names, dict):
                self._names = {int(key): str(value) for key, value in names.items()}
            self.available = True
            self.status_message = (
                f"Object detector ready: ultralytics model={path}, "
                f"confidence>={self.confidence_threshold:.2f}, "
                f"device={self.device} requested={self.requested_device} "
                f"backend={self.accelerator_status.backend}"
            )
        except Exception as exc:
            self.status_message = f"Object detector failed to load {path}: {exc}"

    def detect(self, frame: Any) -> list[Detection]:
        """Return object detections for a frame."""

        if not self.available or self._model is None:
            return []

        try:
            results = self._model.predict(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            self.available = False
            self.status_message = f"Object detector inference failed; disabling detector: {exc}"
            return []

        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        names = getattr(result, "names", None)
        if isinstance(names, dict):
            label_names = {int(key): str(value) for key, value in names.items()}
        else:
            label_names = self._names

        for box in boxes:
            try:
                xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                confidence = float(box.conf[0].detach().cpu().item())
                class_id = int(box.cls[0].detach().cpu().item())
            except Exception:
                continue

            if confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = (int(round(value)) for value in xyxy[:4])
            detections.append(
                Detection(
                    label=label_names.get(class_id, f"class_{class_id}"),
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    source="object",
                )
            )

        return detections
