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

import numpy as np

from .accelerator import get_torch_accelerator_status
from .camera import Camera, check_camera, cv2
from .config import AppConfig
from .detectors.face_detector import FaceDetector
from .detectors.object_detector import Detection, ObjectDetector
from .scene_state import SceneState, build_scene_state


WINDOW_NAME = "Local AI Vision Assistant"


def _viewer_window_flags() -> int:
    """Return OpenCV window flags for a resizable viewer without Qt toolbars."""

    flags = cv2.WINDOW_NORMAL
    gui_normal = getattr(cv2, "WINDOW_GUI_NORMAL", None)
    if gui_normal is not None:
        flags |= int(gui_normal)
    return flags


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _check_torch() -> list[str]:
    status = get_torch_accelerator_status("auto")
    if not status.torch_available:
        return [
            "torch: not installed",
            "ROCm/HIP: unavailable because torch is not installed",
        ]

    lines = [f"torch: available ({status.torch_version})"]
    lines.append(f"ROCm/HIP version: {status.hip_version or 'not reported by torch'}")
    lines.append(f"CUDA build version: {status.cuda_version or 'not reported by torch'}")
    lines.append(f"torch GPU available: {status.gpu_available}")
    lines.append(f"torch GPU device count: {status.device_count}")
    lines.append(f"torch GPU device name: {status.device_name or 'none'}")
    lines.append(f"torch auto device: {status.resolved_device} ({status.note})")
    return lines


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
    accelerator = get_torch_accelerator_status(config.object_device)
    lines = [
        f"Object detector backend: {config.object_detector_backend}",
        f"Object detector model: {config.object_model_path}",
        f"Object confidence threshold: {config.object_confidence_threshold:.2f}",
        f"Object detection interval: every {config.object_detection_interval} frame(s)",
        f"Object detector requested device: {config.object_device}",
        f"Object detector resolved device: {accelerator.resolved_device}",
        f"Object accelerator backend: {accelerator.backend}",
        f"Object accelerator note: {accelerator.note}",
    ]
    model_path = Path(config.object_model_path)
    lines.append(f"Object model file: {'found' if model_path.exists() else 'missing'}")
    lines.append(
        "Ultralytics: available"
        if _module_available("ultralytics")
        else "Ultralytics: not installed"
    )
    return lines


def _check_face_detector(config: AppConfig) -> list[str]:
    lines = [
        f"Face detection enabled: {config.enable_face_detection}",
        f"Face detector model: {config.face_model_path}",
    ]
    try:
        detector = FaceDetector(model_path=config.face_model_path)
        lines.append(f"Face detector available: {detector.available}")
        lines.append(f"Face detector status: {detector.status_message}")
    except Exception as exc:
        lines.append(f"Face detector check failed: {exc}")
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

    camera_status = check_camera(config.camera_index, resolution=config.camera_resolution)
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

    for line in _check_face_detector(config):
        print(line)

    print(_check_vllm(config))
    print()
    print("Configuration")
    print("-" * 13)
    print(f"camera_index: {config.camera_index}")
    print(f"target_fps: {config.target_fps}")
    print(f"camera_resolution_mode: {config.camera_resolution_mode}")
    print(f"camera_resolution: {config.camera_resolution[0]}x{config.camera_resolution[1]}")
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
    print(f"scene_state_interval_seconds: {config.scene_state_interval_seconds}")
    print(f"scene_state_log_path: {config.scene_state_log_path}")
    return 0


def _draw_text(
    frame: Any,
    text: str,
    x: int,
    y: int,
    colour: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.55,
    shadow: bool = True,
) -> None:
    """Draw readable text without thick outline artefacts when the window scales."""

    if shadow:
        cv2.putText(
            frame,
            text,
            (x + 1, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        1,
        cv2.LINE_AA,
    )


def _truncate_text(text: str, max_chars: int) -> str:
    """Return text shortened for fixed-width OpenCV panels."""

    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _draw_card(
    frame: Any,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    fill: tuple[int, int, int] = (31, 34, 42),
    border: tuple[int, int, int] = (58, 64, 78),
) -> None:
    """Draw a simple dashboard card."""

    cv2.rectangle(frame, (x, y), (x + width, y + height), fill, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), border, 1)


def _draw_chip(
    frame: Any,
    text: str,
    x: int,
    y: int,
    *,
    active: bool,
    accent: tuple[int, int, int],
) -> int:
    """Draw a compact status chip and return its width."""

    font_scale = 0.46
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    width = text_width + 22
    height = text_height + 14
    fill = accent if active else (54, 58, 68)
    text_colour = (18, 22, 26) if active else (210, 214, 222)
    cv2.rectangle(frame, (x, y), (x + width, y + height), fill, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (86, 92, 108), 1)
    cv2.putText(
        frame,
        text,
        (x + 11, y + height - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_colour,
        1,
    )
    return width


def _save_frame(frame: Any, captures_dir: str) -> Path:
    Path(captures_dir).mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f.jpg")
    path = Path(captures_dir) / filename
    cv2.imwrite(str(path), frame)
    return path


def _append_scene_state_log(line: str, log_path: str) -> None:
    """Append one JSONL scene-state snapshot to an explicit log path."""

    if not log_path:
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def _emit_scene_state(state: SceneState, log_path: str) -> None:
    """Print a scene-state JSONL line and optionally append it to a file."""

    line = state.to_jsonl()
    print(line, flush=True)
    if log_path:
        try:
            _append_scene_state_log(line, log_path)
        except OSError as exc:
            print(f"Warning: could not write scene-state log {log_path!r}: {exc}")


def _scene_state_interval_due(
    *,
    now: float,
    last_emit_time: float | None,
    interval_seconds: float,
) -> bool:
    """Return whether interval-based scene-state output is due."""

    if interval_seconds <= 0:
        return False
    return last_emit_time is None or now - last_emit_time >= interval_seconds


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


def _draw_face_detections(frame: Any, detections: list[Detection]) -> None:
    """Draw generic non-identifying face boxes onto the current frame."""

    height, width = frame.shape[:2]
    face_index = 0
    for detection in detections:
        if detection.source != "face":
            continue

        x1, y1, x2, y2 = detection.bbox
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        face_index += 1
        colour = (255, 190, 90)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"face {face_index}"
        cv2.putText(
            frame,
            label,
            (x1, max(16, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
        )


def _blur_face_regions(frame: Any, detections: list[Detection]) -> None:
    """Blur detected face regions in-place without storing face data."""

    height, width = frame.shape[:2]
    for detection in detections:
        if detection.source != "face":
            continue

        x1, y1, x2, y2 = detection.bbox
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        region = frame[y1:y2, x1:x2]
        blur_width = max(15, ((x2 - x1) // 2) | 1)
        blur_height = max(15, ((y2 - y1) // 2) | 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (blur_width, blur_height), 0)


def _should_draw_face_boxes(face_detection_enabled: bool) -> bool:
    """Return whether generic face boxes should be drawn over the frame."""

    return face_detection_enabled


def _draw_overlay(
    frame: Any,
    fps: float,
    frame_id: int,
    config: AppConfig,
    face_detection_enabled: bool,
    face_detections: list[Detection],
    face_status: str,
    object_detection_enabled: bool,
    object_detections: list[Detection],
    object_status: str,
    privacy_blur: bool,
    show_help: bool,
    last_capture: str | None,
) -> Any:
    """Compose a dashboard-style viewer frame around the live camera image."""

    height, width = frame.shape[:2]
    sidebar_width = 340
    padding = 24
    header_height = 84
    footer_height = 74 if show_help else 42
    content_top = header_height + padding
    minimum_canvas_width = 1040
    minimum_canvas_height = 620
    canvas_width = max(minimum_canvas_width, width + sidebar_width + padding * 3)
    canvas_height = max(minimum_canvas_height, height + header_height + footer_height + padding * 2)
    canvas = np.full((canvas_height, canvas_width, 3), (18, 20, 24), dtype=frame.dtype)

    # Header
    cv2.rectangle(canvas, (0, 0), (canvas_width, header_height), (28, 31, 38), -1)
    cv2.line(canvas, (0, header_height - 1), (canvas_width, header_height - 1), (58, 64, 78), 1)
    _draw_text(canvas, "Local AI Vision Assistant", padding, 34, (245, 247, 250), 0.72)
    _draw_text(
        canvas,
        f"MVP 4 · {config.camera_resolution_mode} camera mode · local scene-state JSON · no identity recognition",
        padding,
        60,
        (154, 163, 178),
        0.48,
    )

    chip_y = 24
    chip_x = canvas_width - padding
    chips = [
        ("JSON", config.scene_state_interval_seconds > 0, (120, 220, 255)),
        ("Privacy", privacy_blur, (255, 190, 90)),
        ("Faces", face_detection_enabled, (180, 255, 180)),
        ("Objects", object_detection_enabled, (120, 220, 255)),
    ]
    for text, active, colour in chips:
        chip_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0] + 22
        chip_x -= chip_width
        _draw_chip(canvas, text, chip_x, chip_y, active=active, accent=colour)
        chip_x -= 8

    # Main content layout.
    content_height = canvas_height - content_top - footer_height - padding
    video_area_width = canvas_width - sidebar_width - padding * 3
    video_scale = min(video_area_width / width, content_height / height)
    resized_width = max(1, int(width * video_scale))
    resized_height = max(1, int(height * video_scale))
    if resized_width != width or resized_height != height:
        video = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    else:
        video = frame

    video_card_x = padding
    video_card_y = content_top
    video_card_width = video_area_width
    video_card_height = content_height
    _draw_card(canvas, video_card_x, video_card_y, video_card_width, video_card_height)
    _draw_text(canvas, "LIVE VIEW", video_card_x + 18, video_card_y + 28, (154, 163, 178), 0.45)
    _draw_text(
        canvas,
        f"{width}x{height} actual · {config.camera_resolution_mode} mode · {fps:4.1f} FPS · frame {frame_id}",
        video_card_x + 18,
        video_card_y + 52,
        (230, 234, 242),
        0.52,
    )

    video_x = video_card_x + max(18, (video_card_width - resized_width) // 2)
    video_y = video_card_y + max(66, (video_card_height - resized_height + 52) // 2)
    canvas[video_y : video_y + resized_height, video_x : video_x + resized_width] = video
    cv2.rectangle(
        canvas,
        (video_x - 1, video_y - 1),
        (video_x + resized_width + 1, video_y + resized_height + 1),
        (82, 92, 112),
        1,
    )

    # Sidebar inspector.
    panel_x = canvas_width - sidebar_width - padding
    panel_y = content_top
    _draw_card(canvas, panel_x, panel_y, sidebar_width, content_height)
    _draw_text(canvas, "SCENE SUMMARY", panel_x + 18, panel_y + 30, (154, 163, 178), 0.45)

    counts = _object_counts(object_detections)
    count_text = ", ".join(f"{label}:{count}" for label, count in counts.most_common(4))
    if not count_text:
        count_text = "none"
    face_status_short = _truncate_text(face_status, 38)
    object_status_short = _truncate_text(object_status, 38)

    y = panel_y + 62
    _draw_text(
        canvas,
        f"Objects {'ON' if object_detection_enabled else 'OFF'}",
        panel_x + 18,
        y,
        (180, 255, 180) if object_detection_enabled else (230, 230, 230),
        0.58,
    )
    _draw_text(canvas, f"{len(object_detections)} detections · {count_text}", panel_x + 18, y + 26, (214, 219, 229), 0.48)

    y += 68
    _draw_text(
        canvas,
        f"Faces {'ON' if face_detection_enabled else 'OFF'}",
        panel_x + 18,
        y,
        (180, 255, 180) if face_detection_enabled else (230, 230, 230),
        0.58,
    )
    _draw_text(canvas, f"{len(face_detections)} generic faces · blur {'ON' if privacy_blur else 'OFF'}", panel_x + 18, y + 26, (214, 219, 229), 0.48)

    y += 72
    cv2.line(canvas, (panel_x + 18, y - 18), (panel_x + sidebar_width - 18, y - 18), (58, 64, 78), 1)
    _draw_text(canvas, "PRIVACY", panel_x + 18, y, (154, 163, 178), 0.45)
    _draw_text(canvas, "Identity recognition: disabled", panel_x + 18, y + 28, (180, 255, 180), 0.48)
    _draw_text(canvas, "Face embeddings: not stored", panel_x + 18, y + 52, (180, 255, 180), 0.48)

    y += 98
    cv2.line(canvas, (panel_x + 18, y - 18), (panel_x + sidebar_width - 18, y - 18), (58, 64, 78), 1)
    _draw_text(canvas, "STATUS", panel_x + 18, y, (154, 163, 178), 0.45)
    _draw_text(canvas, f"Object: {object_status_short}", panel_x + 18, y + 28, (120, 220, 255), 0.43)
    _draw_text(canvas, f"Face: {face_status_short}", panel_x + 18, y + 52, (255, 220, 160), 0.43)

    y += 98
    cv2.line(canvas, (panel_x + 18, y - 18), (panel_x + sidebar_width - 18, y - 18), (58, 64, 78), 1)
    _draw_text(canvas, "ACTIONS", panel_x + 18, y, (154, 163, 178), 0.45)
    _draw_text(canvas, "j  emit scene JSONL", panel_x + 18, y + 28, (230, 234, 242), 0.48)
    _draw_text(canvas, "s  save displayed frame", panel_x + 18, y + 52, (230, 234, 242), 0.48)
    _draw_text(canvas, "h  compact help", panel_x + 18, y + 76, (230, 234, 242), 0.48)

    # Footer controls.
    footer_y = canvas_height - footer_height
    cv2.rectangle(canvas, (0, footer_y), (canvas_width, canvas_height), (28, 31, 38), -1)
    cv2.line(canvas, (0, footer_y), (canvas_width, footer_y), (58, 64, 78), 1)
    footer_text = (
        "q quit   ·   f face detection   ·   o object detection   ·   p privacy blur"
        if show_help
        else "Press h for controls"
    )
    _draw_text(canvas, footer_text, padding, footer_y + 28, (214, 219, 229), 0.5)
    _draw_text(
        canvas,
        f"Scene JSON: {'interval ' + str(config.scene_state_interval_seconds) + 's' if config.scene_state_interval_seconds > 0 else 'press j'}",
        padding,
        footer_y + 52 if show_help else footer_y + 28,
        (120, 220, 255),
        0.46,
    )

    if last_capture:
        _draw_text(
            canvas,
            _truncate_text(f"Saved: {last_capture}", 58),
            canvas_width - 430,
            footer_y + 28,
            (160, 255, 160),
            0.46,
        )

    return canvas


def run_viewer(config: AppConfig) -> int:
    """Run the OpenCV webcam viewer."""

    if cv2 is None:
        print("OpenCV is not installed. Install dependencies with:")
        print("  python -m pip install -r requirements.txt")
        return 1

    camera = Camera(
        index=config.camera_index,
        target_fps=config.target_fps,
        resolution=config.camera_resolution,
    )
    status = camera.open()
    if not status.available:
        print(status.message)
        print("Try setting VISION_CAMERA_INDEX to another value, for example:")
        print("  VISION_CAMERA_INDEX=1 python -m src.main run")
        return 1

    print(status.message)
    face_detector = FaceDetector(model_path=config.face_model_path)
    print(face_detector.status_message)
    detector = ObjectDetector(
        model_path=config.object_model_path,
        backend=config.object_detector_backend,
        confidence_threshold=config.object_confidence_threshold,
        device=config.object_device,
    )
    print(detector.status_message)
    print("Keyboard controls: q quit, s save, j scene JSON, f face detection, o object detection, p privacy blur, h help")

    cv2.namedWindow(WINDOW_NAME, _viewer_window_flags())

    frame_id = 0
    fps = 0.0
    last_time = time.perf_counter()
    last_capture: str | None = None
    face_detection_enabled = config.enable_face_detection
    face_detections: list[Detection] = []
    object_detection_enabled = config.enable_object_detection
    object_detections: list[Detection] = []
    privacy_blur = False
    show_help = True
    frame_delay = 1.0 / max(1, config.target_fps)
    last_scene_state_emit: float | None = None

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

            if (face_detection_enabled or privacy_blur) and face_detector.available:
                face_detections = face_detector.detect(frame)
            elif not face_detection_enabled and not privacy_blur:
                face_detections = []

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

            if privacy_blur:
                _blur_face_regions(frame, face_detections)
            if _should_draw_face_boxes(face_detection_enabled):
                _draw_face_detections(frame, face_detections)

            _draw_object_detections(frame, object_detections)

            display_frame = _draw_overlay(
                frame,
                fps,
                frame_id,
                config,
                face_detection_enabled,
                face_detections,
                face_detector.status_message,
                object_detection_enabled,
                object_detections,
                detector.status_message,
                privacy_blur,
                show_help,
                last_capture,
            )

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = _save_frame(display_frame, config.captures_dir)
                last_capture = str(path)
                print(f"Saved frame to {path}")
            elif key == ord("f"):
                face_detection_enabled = not face_detection_enabled
                if face_detection_enabled:
                    print(f"Face detection: ON - {face_detector.status_message}")
                    if not face_detector.available:
                        print("Face detection is enabled, but no usable detector is available.")
                else:
                    face_detections = []
                    print("Face detection: OFF")
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
                print(f"Privacy blur: {'ON' if privacy_blur else 'OFF'}")
                if privacy_blur and not face_detector.available:
                    print("Privacy blur needs face detection, but no usable detector is available.")
            elif key == ord("h"):
                show_help = not show_help

            interval_check_time = time.perf_counter()
            should_emit_scene_state = key == ord("j") or _scene_state_interval_due(
                now=interval_check_time,
                last_emit_time=last_scene_state_emit,
                interval_seconds=config.scene_state_interval_seconds,
            )
            if should_emit_scene_state:
                state = build_scene_state(
                    frame_id=frame_id,
                    fps=fps,
                    object_detections=object_detections,
                    face_detections=face_detections,
                    face_detection_enabled=face_detection_enabled,
                    privacy_blur_enabled=privacy_blur,
                )
                _emit_scene_state(state, config.scene_state_log_path)
                last_scene_state_emit = interval_check_time

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
