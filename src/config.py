"""Application configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os

CAMERA_RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "fast": (640, 480),
    "quality": (1280, 720),
}


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


def _env_csv(name: str, default: tuple[str, ...] = ()) -> list[str]:
    """Read a comma-separated environment variable as trimmed values."""

    value = os.getenv(name)
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    """Read and normalise an environment variable constrained to known choices."""

    value = os.getenv(name)
    if value is None:
        return default
    normalised = value.strip().lower()
    if normalised in choices:
        return normalised
    print(f"Warning: {name}={value!r} is unsupported; using {default}.")
    return default


@dataclass(slots=True)
class AppConfig:
    """Runtime settings for the local vision demo."""

    camera_index: int = 0
    target_fps: int = 30
    camera_resolution_mode: str = "fast"
    enable_object_detection: bool = False
    enable_face_detection: bool = False
    enable_vllm: bool = False
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "local-model"
    object_model_path: str = "models/yolo11n.pt"
    object_detector_backend: str = "ultralytics"
    object_prompts: list[str] | None = None
    object_confidence_threshold: float = 0.35
    object_detection_interval: int = 3
    object_detection_hold_frames: int = 8
    object_device: str = "cpu"
    face_model_path: str = "models/face_detection_yunet_2026may.onnx"
    captures_dir: str = "captures"
    logs_dir: str = "logs"
    scene_state_interval_seconds: float = 0.0
    scene_state_log_path: str = ""

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""

        return cls(
            camera_index=_env_int("VISION_CAMERA_INDEX", 0),
            target_fps=max(1, _env_int("VISION_TARGET_FPS", 30)),
            camera_resolution_mode=_env_choice(
                "VISION_CAMERA_RESOLUTION_MODE",
                "fast",
                set(CAMERA_RESOLUTION_PRESETS),
            ),
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
            object_prompts=_env_csv("VISION_OBJECT_PROMPTS"),
            object_confidence_threshold=min(
                1.0, max(0.0, _env_float("VISION_OBJECT_CONFIDENCE_THRESHOLD", 0.35))
            ),
            object_detection_interval=max(1, _env_int("VISION_OBJECT_DETECTION_INTERVAL", 3)),
            object_detection_hold_frames=max(
                0, _env_int("VISION_OBJECT_DETECTION_HOLD_FRAMES", 8)
            ),
            object_device=os.getenv("VISION_OBJECT_DEVICE", "cpu").strip() or "cpu",
            face_model_path=os.getenv(
                "VISION_FACE_MODEL_PATH", "models/face_detection_yunet_2026may.onnx"
            ),
            captures_dir=os.getenv("VISION_CAPTURES_DIR", "captures"),
            logs_dir=os.getenv("VISION_LOGS_DIR", "logs"),
            scene_state_interval_seconds=max(
                0.0, _env_float("VISION_SCENE_STATE_INTERVAL_SECONDS", 0.0)
            ),
            scene_state_log_path=os.getenv("VISION_SCENE_STATE_LOG_PATH", "").strip(),
        )

    @property
    def camera_resolution(self) -> tuple[int, int]:
        """Return the selected camera capture resolution as ``(width, height)``."""

        return CAMERA_RESOLUTION_PRESETS.get(
            self.camera_resolution_mode,
            CAMERA_RESOLUTION_PRESETS["fast"],
        )

    @property
    def active_modes(self) -> str:
        """Return a compact description of configured optional modes."""

        camera_mode = f"camera:{self.camera_resolution_mode}"
        object_mode = (
            f"objects:on/{self.object_detector_backend}"
            if self.enable_object_detection
            else "objects:off"
        )
        face_mode = "faces:on" if self.enable_face_detection else "faces:off"
        vllm_mode = "vLLM:on" if self.enable_vllm else "vLLM:off"
        return f"{camera_mode} | {object_mode} | {face_mode} | {vllm_mode}"
