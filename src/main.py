"""Command-line entry point for the Local AI Vision Assistant MVP."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import importlib.util
from pathlib import Path
import platform
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .camera import Camera, check_camera, cv2
from .config import AppConfig
from .detectors.object_detector import Detection, ObjectDetector


WINDOW_NAME = "Local AI Vision Assistant"


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _check_torch() -> list[str]:
    if not _module_available("torch"):
        return ["torch: not installed", "ROCm/HIP: unavailable because torch is not installed"]

    try:
        import torch

        lines = [f"torch: available ({torch.__version__})"]
        hip_version = getattr(getattr(torch, "version", None), "hip", None)
        lines.append(f"ROCm/HIP version: {hip_version or 'not reported by torch'}")
        try:
            lines.append(f"torch GPU available: {torch.cuda.is_available()}")
            lines.append(f"torch GPU device count: {torch.cuda.device_count()}")
        except Exception as exc:  # pragma: no cover - hardware-dependent.
            lines.append(f"torch GPU check warning: {exc}")
        return lines
    except Exception as exc:
        return [f"torch: import failed ({exc})", "ROCm/HIP: unavailable because torch failed"]


def _check_vllm(config: AppConfig) -> str:
    if not config.enable_vllm:
        return "vLLM endpoint: skipped (VISION_ENABLE_VLLM is false)"

    url = f"{config.vllm_base_url.rstrip('/')}/models"
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=2) as response:
            return f"vLLM endpoint: reachable ({response.status} from {url})"
    except URLError as exc:
        return f"vLLM endpoint: unavailable at {url} ({exc})"
    except Exception as exc:
        return f"vLLM endpoint: check failed at {url} ({exc})"


def _check_object_detector(config: AppConfig) -> list[str]:
    lines = [
        f"Object detector backend: {config.object_detector_backend}",
        f"Object detector model: {config.object_model_path}",
        f"Object confidence threshold: {config.object_confidence_threshold:.2f}",
        f"Object detection interval: every {config.object_detection_interval} frame(s)",
        f"Object detector device: {config.object_device}",
    ]
    model_path = Path(config.object_model_path)
    lines.append(f"Object model file: {'found' if model_path.exists() else 'missing'}")
    lines.append(
        "Ultralytics: available"
        if _module_available("ultralytics")
        else "Ultralytics: not installed"
    )
    return lines


def run_health_check(config: AppConfig) -> int:
    """Print local environment and camera health information."""

    print("Local AI Vision Assistant health check")
    print("=" * 45)
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")

    if cv2 is None:
        print("OpenCV: not installed")
    else:
        print(f"OpenCV: available ({cv2.__version__})")

    camera_status = check_camera(config.camera_index)
    camera_label = "available" if camera_status.available else "unavailable"
    print(f"Camera: {camera_label} - {camera_status.message}")

    for line in _check_torch():
        print(line)

    if _module_available("onnxruntime"):
        try:
            import onnxruntime as ort

            print(f"ONNX Runtime: available ({ort.__version__})")
            print(f"ONNX Runtime providers: {', '.join(ort.get_available_providers())}")
        except Exception as exc:
            print(f"ONNX Runtime: import failed ({exc})")
    else:
        print("ONNX Runtime: not installed")

    for line in _check_object_detector(config):
        print(line)

    print(_check_vllm(config))
    print()
    print("Configuration")
    print("-" * 13)
    print(f"camera_index: {config.camera_index}")
    print(f"target_fps: {config.target_fps}")
    print(f"enable_object_detection: {config.enable_object_detection}")
    print(f"enable_face_detection: {config.enable_face_detection}")
    print(f"enable_vllm: {config.enable_vllm}")
    print(f"vllm_base_url: {config.vllm_base_url}")
    print(f"vllm_model: {config.vllm_model}")
    print(f"object_model_path: {config.object_model_path}")
    print(f"object_detector_backend: {config.object_detector_backend}")
    print(f"object_confidence_threshold: {config.object_confidence_threshold}")
    print(f"object_detection_interval: {config.object_detection_interval}")
    print(f"object_device: {config.object_device}")
    print(f"face_model_path: {config.face_model_path}")
    return 0


def _draw_text(
    frame: Any,
    text: str,
    x: int,
    y: int,
    colour: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.55,
) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1)


def _save_frame(frame: Any, captures_dir: str) -> Path:
    Path(captures_dir).mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f.jpg")
    path = Path(captures_dir) / filename
    cv2.imwrite(str(path), frame)
    return path


def _object_counts(detections: list[Detection]) -> Counter[str]:
    return Counter(detection.label for detection in detections if detection.source == "object")


def _draw_object_detections(frame: Any, detections: list[Detection]) -> None:
    """Draw object boxes and labels onto the current frame."""

    height, width = frame.shape[:2]
    for detection in detections:
        if detection.source != "object":
            continue

        x1, y1, x2, y2 = detection.bbox
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        colour = (80, 220, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"{detection.label} {detection.confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y1 = max(0, y1 - text_height - 8)
        label_y2 = max(text_height + 8, y1)
        cv2.rectangle(frame, (x1, label_y1), (x1 + text_width + 8, label_y2), colour, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )


def _draw_overlay(
    frame: Any,
    fps: float,
    frame_id: int,
    config: AppConfig,
    show_face_placeholder: bool,
    object_detection_enabled: bool,
    object_detections: list[Detection],
    object_status: str,
    privacy_blur: bool,
    show_help: bool,
    last_capture: str | None,
) -> None:
    height, width = frame.shape[:2]
    panel_height = 188 if show_help else 150
    cv2.rectangle(frame, (0, 0), (width, panel_height), (20, 20, 20), -1)
    _draw_text(frame, "Local AI Vision Assistant - MVP 2 Object Detection", 12, 24, (120, 220, 255), 0.6)
    _draw_text(frame, f"FPS: {fps:5.1f} | Frame: {frame_id} | Size: {width}x{height}", 12, 48)
    _draw_text(frame, f"Modes: {config.active_modes}", 12, 72)
    counts = _object_counts(object_detections)
    count_text = ", ".join(f"{label}:{count}" for label, count in counts.most_common(4))
    if not count_text:
        count_text = "none"
    object_status_short = object_status if len(object_status) <= 92 else object_status[:89] + "..."
    _draw_text(
        frame,
        f"Objects: {'ON' if object_detection_enabled else 'OFF'} | "
        f"detections {len(object_detections)} | counts {count_text}",
        12,
        96,
        (180, 255, 180) if object_detection_enabled else (230, 230, 230),
        0.5,
    )
    _draw_text(
        frame,
        f"Object status: {object_status_short}",
        12,
        120,
        (180, 255, 180) if "ready" in object_status.lower() else (120, 220, 255),
        0.5,
    )
    _draw_text(
        frame,
        f"Face placeholder {'ON' if show_face_placeholder else 'OFF'} | "
        f"privacy blur placeholder {'ON' if privacy_blur else 'OFF'}",
        12,
        144,
        (180, 255, 180) if privacy_blur else (230, 230, 230),
        0.48,
    )

    if show_help:
        _draw_text(frame, "Keys: q quit | s save | f face placeholder | o object detection | p privacy blur | h help", 12, 168, (220, 220, 255), 0.48)

    if last_capture:
        _draw_text(frame, f"Saved: {last_capture}", 12, height - 16, (160, 255, 160), 0.5)


def run_viewer(config: AppConfig) -> int:
    """Run the OpenCV webcam viewer."""

    if cv2 is None:
        print("OpenCV is not installed. Install dependencies with:")
        print("  python -m pip install -r requirements.txt")
        return 1

    camera = Camera(index=config.camera_index, target_fps=config.target_fps)
    status = camera.open()
    if not status.available:
        print(status.message)
        print("Try setting VISION_CAMERA_INDEX to another value, for example:")
        print("  VISION_CAMERA_INDEX=1 python -m src.main run")
        return 1

    print(status.message)
    detector = ObjectDetector(
        model_path=config.object_model_path,
        backend=config.object_detector_backend,
        confidence_threshold=config.object_confidence_threshold,
        device=config.object_device,
    )
    print(detector.status_message)
    print("Keyboard controls: q quit, s save, f face placeholder, o object detection, p privacy blur, h help")

    frame_id = 0
    fps = 0.0
    last_time = time.perf_counter()
    last_capture: str | None = None
    show_face_placeholder = config.enable_face_detection
    object_detection_enabled = config.enable_object_detection
    object_detections: list[Detection] = []
    privacy_blur = False
    show_help = True
    frame_delay = 1.0 / max(1, config.target_fps)

    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera frame read failed; exiting.")
                return 1

            frame_id += 1
            now = time.perf_counter()
            elapsed = now - last_time
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = current_fps if fps == 0.0 else (fps * 0.9 + current_fps * 0.1)
            last_time = now

            if object_detection_enabled and detector.available:
                should_infer = (
                    frame_id == 1
                    or frame_id % config.object_detection_interval == 0
                    or not object_detections
                )
                if should_infer:
                    object_detections = detector.detect(frame)
            elif not object_detection_enabled:
                object_detections = []

            _draw_object_detections(frame, object_detections)

            _draw_overlay(
                frame,
                fps,
                frame_id,
                config,
                show_face_placeholder,
                object_detection_enabled,
                object_detections,
                detector.status_message,
                privacy_blur,
                show_help,
                last_capture,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = _save_frame(frame, config.captures_dir)
                last_capture = str(path)
                print(f"Saved frame to {path}")
            elif key == ord("f"):
                show_face_placeholder = not show_face_placeholder
                print(f"Face overlay placeholder: {'ON' if show_face_placeholder else 'OFF'}")
            elif key == ord("o"):
                object_detection_enabled = not object_detection_enabled
                if object_detection_enabled:
                    print(f"Object detection: ON - {detector.status_message}")
                    if not detector.available:
                        print("Object detection is enabled, but no usable detector is available.")
                else:
                    object_detections = []
                    print("Object detection: OFF")
            elif key == ord("p"):
                privacy_blur = not privacy_blur
                print(f"Privacy blur placeholder: {'ON' if privacy_blur else 'OFF'}")
            elif key == ord("h"):
                show_help = not show_help

            spent = time.perf_counter() - loop_start
            if spent < frame_delay:
                time.sleep(frame_delay - spent)
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Local AI Vision Assistant MVP")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("health", help="Print environment and camera health information")
    subparsers.add_parser("run", help="Open the webcam viewer")
    subparsers.add_parser("config", help="Print the resolved configuration")
    return parser


def print_config(config: AppConfig) -> int:
    """Print configuration values."""

    for key, value in asdict(config).items():
        print(f"{key}: {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run a CLI command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    config = AppConfig.from_env()

    command = args.command or "run"
    if command == "health":
        return run_health_check(config)
    if command == "run":
        return run_viewer(config)
    if command == "config":
        return print_config(config)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
