"""OpenCV camera helpers for the local vision demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:
    import cv2
except ImportError:  # pragma: no cover - exercised only when OpenCV is absent.
    cv2 = None  # type: ignore[assignment]


@dataclass(slots=True)
class CameraStatus:
    """Result from checking or opening a camera."""

    available: bool
    message: str


class Camera:
    """Small wrapper around ``cv2.VideoCapture``."""

    def __init__(self, index: int = 0, target_fps: int = 30) -> None:
        self.index = index
        self.target_fps = target_fps
        self.capture: Any | None = None

    def open(self) -> CameraStatus:
        """Open the camera if OpenCV is available."""

        if cv2 is None:
            return CameraStatus(False, "OpenCV is not installed.")

        self.capture = cv2.VideoCapture(self.index)
        if self.capture is None or not self.capture.isOpened():
            self.release()
            return CameraStatus(False, f"Camera index {self.index} could not be opened.")

        self.capture.set(cv2.CAP_PROP_FPS, float(self.target_fps))
        return CameraStatus(True, f"Camera index {self.index} opened.")

    def read(self) -> tuple[bool, Any | None]:
        """Read one frame from the camera."""

        if self.capture is None:
            return False, None
        ok, frame = self.capture.read()
        return bool(ok), frame

    def release(self) -> None:
        """Release the camera if it is open."""

        if self.capture is not None:
            self.capture.release()
            self.capture = None


def check_camera(index: int = 0) -> CameraStatus:
    """Return whether a camera can be opened, without raising on failure."""

    camera = Camera(index=index)
    status = camera.open()
    camera.release()
    return status
