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
        "Object detector prompts: "
        + (", ".join(config.object_prompts or []) if config.object_prompts else "none"),
        f"Object confidence threshold: {config.object_confidence_threshold:.2f}",
        f"Object detection interval: every {config.object_detection_interval} frame(s)",
        f"Object detection hold: {config.object_detection_hold_frames} frame(s)",
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
    print(f"object_prompts: {', '.join(config.object_prompts or [])}")
    print(f"object_confidence_threshold: {config.object_confidence_threshold}")
    print(f"object_detection_interval: {config.object_detection_interval}")
    print(f"object_detection_hold_frames: {config.object_detection_hold_frames}")
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


def _draw_rounded_box(
    frame: Any,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    radius: int = 14,
    fill: tuple[int, int, int] = (31, 34, 42),
    border: tuple[int, int, int] | None = (58, 64, 78),
    alpha: float = 1.0,
) -> None:
    """Draw a rounded rectangle with optional alpha blending."""

    radius = max(0, min(radius, width // 2, height // 2))
    target = frame
    overlay = frame.copy() if alpha < 1.0 else frame

    # Filled rounded box.
    cv2.rectangle(overlay, (x + radius, y), (x + width - radius, y + height), fill, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + width, y + height - radius), fill, -1)
    cv2.circle(overlay, (x + radius, y + radius), radius, fill, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + width - radius, y + radius), radius, fill, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + radius, y + height - radius), radius, fill, -1, cv2.LINE_AA)
    cv2.circle(
        overlay,
        (x + width - radius, y + height - radius),
        radius,
        fill,
        -1,
        cv2.LINE_AA,
    )

    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, target, 1.0 - alpha, 0, target)

    if border is not None:
        cv2.rectangle(
            frame,
            (x + radius, y),
            (x + width - radius, y + height),
            border,
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            frame,
            (x, y + radius),
            (x + width, y + height - radius),
            border,
            1,
            cv2.LINE_AA,
        )
        cv2.ellipse(frame, (x + radius, y + radius), (radius, radius), 180, 0, 90, border, 1, cv2.LINE_AA)
        cv2.ellipse(frame, (x + width - radius, y + radius), (radius, radius), 270, 0, 90, border, 1, cv2.LINE_AA)
        cv2.ellipse(frame, (x + radius, y + height - radius), (radius, radius), 90, 0, 90, border, 1, cv2.LINE_AA)
        cv2.ellipse(
            frame,
            (x + width - radius, y + height - radius),
            (radius, radius),
            0,
            0,
            90,
            border,
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
    """Draw a polished dashboard card."""

    _draw_rounded_box(
        frame,
        x,
        y,
        width,
        height,
        radius=18,
        fill=fill,
        border=border,
        alpha=0.96,
    )


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
    _draw_rounded_box(
        frame,
        x,
        y,
        width,
        height,
        radius=height // 2,
        fill=fill,
        border=(86, 92, 108),
        alpha=0.94,
    )
    cv2.putText(
        frame,
        text,
        (x + 11, y + height - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_colour,
        1,
        cv2.LINE_AA,
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


def _wrap_text(text: str, max_width: int, scale: float = 0.45) -> list[str]:
    """Wrap text into OpenCV-renderable lines that fit a pixel width."""

    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        text_width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if text_width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_text_block(
    frame: Any,
    lines: list[str],
    x: int,
    y: int,
    max_width: int,
    *,
    colour: tuple[int, int, int] = (214, 219, 229),
    scale: float = 0.45,
    line_height: int = 22,
) -> int:
    """Draw wrapped text lines and return the next y coordinate."""

    cursor_y = y
    for line in lines:
        for wrapped in _wrap_text(line, max_width, scale):
            _draw_text(frame, wrapped, x, cursor_y, colour, scale)
            cursor_y += line_height
    return cursor_y


def _draw_fixed_text_rows(
    frame: Any,
    lines: list[str],
    x: int,
    y: int,
    max_width: int,
    row_count: int,
    *,
    colour: tuple[int, int, int] = (214, 219, 229),
    scale: float = 0.43,
    row_height: int = 24,
) -> None:
    """Draw text into fixed-height rows so the dashboard does not reflow."""

    for index in range(row_count):
        text = lines[index] if index < len(lines) else ""
        wrapped = _wrap_text(text, max_width, scale)
        stable_text = _truncate_text(wrapped[0] if wrapped else "", 64)
        _draw_text(frame, stable_text, x, y + index * row_height, colour, scale)


def _build_demo_commentary(
    *,
    config: AppConfig,
    face_detection_enabled: bool,
    face_detections: list[Detection],
    object_detection_enabled: bool,
    object_detections: list[Detection],
    privacy_blur: bool,
    object_status: str,
) -> list[str]:
    """Return concise live commentary for demo audiences."""

    commentary: list[str] = []
    if object_detection_enabled:
        if object_detections:
            counts = _object_counts(object_detections)
            summary = ", ".join(
                f"{label} x{count}" for label, count in counts.most_common(3)
            )
            commentary.append(f"Object detector is tracking: {summary}.")
        elif "ready" in object_status.lower():
            commentary.append(
                f"Object detector is running every {config.object_detection_interval} frame(s), but has no confident boxes yet."
            )
        else:
            commentary.append("Object detection is enabled, but the selected backend is not ready.")
    else:
        commentary.append("Object detection is paused; press o to enable it.")

    if config.object_detector_backend == "yoloe" or config.object_model_path.lower().endswith("-seg.pt"):
        prompts = ", ".join((config.object_prompts or [])[:5])
        commentary.append(
            f"YOLOE is using text prompts: {prompts}."
            if prompts
            else "YOLOE works best when you provide object prompts."
        )

    if face_detection_enabled:
        commentary.append(
            f"Face mode counts generic faces only: {len(face_detections)} visible."
        )
    elif privacy_blur:
        commentary.append("Face box display is off; blur still uses local face boxes internally.")
    else:
        commentary.append("Face box display is off; press f to show generic face boxes.")

    if privacy_blur:
        commentary.append("Privacy blur is active; face regions are blurred locally before display.")
    else:
        commentary.append("Identity recognition and face embeddings remain disabled.")

    if config.scene_state_interval_seconds > 0:
        commentary.append(
            f"Scene JSONL is emitted every {config.scene_state_interval_seconds:g}s."
        )
    else:
        commentary.append("Press j to emit a compact scene-state JSONL snapshot.")

    return commentary


def _object_counts(detections: list[Detection]) -> Counter[str]:
    return Counter(detection.label for detection in detections if detection.source == "object")


def _detection_colour(label: str) -> tuple[int, int, int]:
    """Return a stable bright BGR colour for a detection label."""

    palette = [
        (80, 220, 120),
        (120, 220, 255),
        (255, 190, 90),
        (190, 150, 255),
        (255, 120, 170),
        (120, 255, 210),
    ]
    index = sum(ord(char) for char in label) % len(palette)
    return palette[index]


def _draw_detection_label(
    frame: Any,
    label: str,
    x: int,
    y: int,
    colour: tuple[int, int, int],
) -> None:
    """Draw a readable semi-transparent detection label."""

    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    box_width = text_width + 12
    box_height = text_height + 12
    label_y = max(0, y - box_height - 6)
    _draw_rounded_box(
        frame,
        x,
        label_y,
        box_width,
        box_height,
        radius=7,
        fill=colour,
        border=None,
        alpha=0.86,
    )
    cv2.putText(
        frame,
        label,
        (x + 6, label_y + box_height - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (10, 12, 16),
        1,
        cv2.LINE_AA,
    )


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

        colour = _detection_colour(detection.label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        corner = max(8, min(24, (x2 - x1) // 6, (y2 - y1) // 6))
        cv2.line(frame, (x1, y1), (x1 + corner, y1), colour, 4, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), colour, 4, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2 - corner, y1), colour, 4, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), colour, 4, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1 + corner, y2), colour, 4, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner), colour, 4, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - corner, y2), colour, 4, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), colour, 4, cv2.LINE_AA)
        label = f"{detection.label} {detection.confidence:.2f}"
        _draw_detection_label(frame, label, x1, y1, colour)


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
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        _draw_detection_label(frame, f"face {face_index}", x1, y1, colour)


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
    sidebar_width = 380
    padding = 24
    header_height = 84
    footer_height = 74 if show_help else 42
    content_top = header_height + padding
    minimum_canvas_width = 1220
    minimum_canvas_height = 740
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
        interpolation = cv2.INTER_CUBIC if resized_width > width else cv2.INTER_AREA
        video = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
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

    commentary_y = panel_y + 208
    cv2.line(
        canvas,
        (panel_x + 18, commentary_y - 18),
        (panel_x + sidebar_width - 18, commentary_y - 18),
        (58, 64, 78),
        1,
    )
    _draw_text(canvas, "DEMO COMMENTARY", panel_x + 18, commentary_y, (154, 163, 178), 0.45)
    commentary = _build_demo_commentary(
        config=config,
        face_detection_enabled=face_detection_enabled,
        face_detections=face_detections,
        object_detection_enabled=object_detection_enabled,
        object_detections=object_detections,
        privacy_blur=privacy_blur,
        object_status=object_status,
    )
    _draw_fixed_text_rows(
        canvas,
        commentary,
        panel_x + 18,
        commentary_y + 30,
        sidebar_width - 36,
        row_count=5,
    )

    privacy_y = panel_y + 382
    cv2.line(
        canvas,
        (panel_x + 18, privacy_y - 18),
        (panel_x + sidebar_width - 18, privacy_y - 18),
        (58, 64, 78),
        1,
    )
    _draw_text(canvas, "PRIVACY", panel_x + 18, privacy_y, (154, 163, 178), 0.45)
    _draw_text(canvas, "Identity recognition: disabled", panel_x + 18, privacy_y + 28, (180, 255, 180), 0.48)
    _draw_text(canvas, "Face embeddings: not stored", panel_x + 18, privacy_y + 52, (180, 255, 180), 0.48)

    status_y = panel_y + 476
    cv2.line(
        canvas,
        (panel_x + 18, status_y - 18),
        (panel_x + sidebar_width - 18, status_y - 18),
        (58, 64, 78),
        1,
    )
    _draw_text(canvas, "STATUS", panel_x + 18, status_y, (154, 163, 178), 0.45)
    _draw_text(canvas, f"Object: {object_status_short}", panel_x + 18, status_y + 28, (120, 220, 255), 0.43)
    _draw_text(canvas, f"Face: {face_status_short}", panel_x + 18, status_y + 52, (255, 220, 160), 0.43)

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
        prompts=config.object_prompts,
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
    last_object_detection_frame: int | None = None
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
                    latest_object_detections = detector.detect(frame)
                    if latest_object_detections:
                        object_detections = latest_object_detections
                        last_object_detection_frame = frame_id
                    elif (
                        object_detections
                        and last_object_detection_frame is not None
                        and frame_id - last_object_detection_frame
                        <= config.object_detection_hold_frames
                    ):
                        # Keep the previous boxes briefly so one weak inference
                        # frame does not make the demo UI flash empty.
                        pass
                    else:
                        object_detections = []
            elif not object_detection_enabled:
                object_detections = []
                last_object_detection_frame = None

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
                    print("Privacy blur needs a usable local face detector, but the face-box display toggle can stay off.")
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
