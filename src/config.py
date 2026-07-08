"""Application configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Warning: {name}={value!r} is not an integer; using {default}.")
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Warning: {name}={value!r} is not a number; using {default}.")
        return default


@dataclass(slots=True)
class AppConfig:
    """Runtime settings for the local vision demo."""

    camera_index: int = 0
    target_fps: int = 30
    enable_object_detection: bool = False
    enable_face_detection: bool = False
    enable_vllm: bool = False
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "local-model"
    object_model_path: str = "models/yolo11n.pt"
    object_detector_backend: str = "ultralytics"
    object_confidence_threshold: float = 0.35
    object_detection_interval: int = 3
    object_device: str = "cpu"
    face_model_path: str = "models/face_detector.onnx"
    captures_dir: str = "captures"
    logs_dir: str = "logs"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""

        return cls(
            camera_index=_env_int("VISION_CAMERA_INDEX", 0),
            target_fps=max(1, _env_int("VISION_TARGET_FPS", 30)),
            enable_object_detection=_env_bool("VISION_ENABLE_OBJECT_DETECTION", False),
            enable_face_detection=_env_bool("VISION_ENABLE_FACE_DETECTION", False),
            enable_vllm=_env_bool("VISION_ENABLE_VLLM", False),
            vllm_base_url=os.getenv("VISION_VLLM_BASE_URL", "http://localhost:8000/v1"),
            vllm_model=os.getenv("VISION_VLLM_MODEL", "local-model"),
            object_model_path=os.getenv(
                "VISION_OBJECT_MODEL_PATH", "models/yolo11n.pt"
            ),
            object_detector_backend=os.getenv(
                "VISION_OBJECT_DETECTOR_BACKEND", "ultralytics"
            ).strip().lower(),
            object_confidence_threshold=min(
                1.0, max(0.0, _env_float("VISION_OBJECT_CONFIDENCE_THRESHOLD", 0.35))
            ),
            object_detection_interval=max(1, _env_int("VISION_OBJECT_DETECTION_INTERVAL", 3)),
            object_device=os.getenv("VISION_OBJECT_DEVICE", "cpu").strip() or "cpu",
            face_model_path=os.getenv("VISION_FACE_MODEL_PATH", "models/face_detector.onnx"),
            captures_dir=os.getenv("VISION_CAPTURES_DIR", "captures"),
            logs_dir=os.getenv("VISION_LOGS_DIR", "logs"),
        )

    @property
    def active_modes(self) -> str:
        """Return a compact description of configured optional modes."""

        object_mode = (
            f"objects:on/{self.object_detector_backend}"
            if self.enable_object_detection
            else "objects:off"
        )
        face_mode = "faces:on" if self.enable_face_detection else "faces:placeholder"
        vllm_mode = "vLLM:on" if self.enable_vllm else "vLLM:off"
        return f"{object_mode} | {face_mode} | {vllm_mode}"
