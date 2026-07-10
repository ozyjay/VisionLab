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

    def __init__(
        self,
        index: int = 0,
        target_fps: int = 30,
        resolution: tuple[int, int] | None = None,
    ) -> None:
        self.index = index
        self.target_fps = target_fps
        self.resolution = resolution
        self.capture: Any | None = None

    def open(self) -> CameraStatus:
        """Open the camera if OpenCV is available."""

        if cv2 is None:
            return CameraStatus(False, "OpenCV is not installed.")

        self.capture = cv2.VideoCapture(self.index)
        if self.capture is None or not self.capture.isOpened():
            self.release()
            return CameraStatus(False, f"Camera index {self.index} could not be opened.")

        if self.resolution is not None:
            width, height = self.resolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        self.capture.set(cv2.CAP_PROP_FPS, float(self.target_fps))

        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        resolution_label = (
            f" at {actual_width}x{actual_height}"
            if actual_width > 0 and actual_height > 0
            else ""
        )
        return CameraStatus(True, f"Camera index {self.index} opened{resolution_label}.")

    def read(self) -> tuple[bool, Any | None]:
        """Read one frame from the camera."""

        if self.capture is None:
            return False, None
        ok, frame = self.capture.read()
        return bool(ok), frame

    def set_target_fps(self, target_fps: int) -> None:
        """Update the requested capture FPS when the backend supports it."""

        self.target_fps = max(1, int(target_fps))
        if self.capture is not None and cv2 is not None:
            self.capture.set(cv2.CAP_PROP_FPS, float(self.target_fps))

    def set_resolution(self, resolution: tuple[int, int]) -> tuple[int, int]:
        """Request a new capture resolution and return the reported dimensions."""

        self.resolution = resolution
        if self.capture is None or cv2 is None:
            return resolution

        width, height = resolution
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
        return actual_width, actual_height

    def release(self) -> None:
        """Release the camera if it is open."""

        if self.capture is not None:
            self.capture.release()
            self.capture = None


def check_camera(
    index: int = 0,
    resolution: tuple[int, int] | None = None,
) -> CameraStatus:
    """Return whether a camera can be opened, without raising on failure."""

    camera = Camera(index=index, resolution=resolution)
    status = camera.open()
    camera.release()
    return status
