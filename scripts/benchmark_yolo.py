"""Benchmark local YOLO inference through the project detector interface."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors.object_detector import ObjectDetector


def _sync_torch() -> None:
    """Synchronise the active Torch GPU, if one is available."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Benchmark YOLO detector inference.")
    parser.add_argument("--model", default="models/yolo11s.pt", help="Local YOLO .pt path.")
    parser.add_argument("--device", default="auto", help="Torch device, for example auto, cpu, rocm, or cuda:0.")
    parser.add_argument("--frames", type=int, default=30, help="Measured frame count.")
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up frame count.")
    parser.add_argument("--width", type=int, default=640, help="Synthetic frame width.")
    parser.add_argument("--height", type=int, default=480, help="Synthetic frame height.")
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence threshold.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark."""

    args = build_parser().parse_args(argv)
    detector = ObjectDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
    )
    print(detector.status_message)
    if not detector.available:
        return 1

    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

    for _ in range(max(0, args.warmup)):
        detector.detect(frame)
    _sync_torch()

    timings_ms: list[float] = []
    for _ in range(max(1, args.frames)):
        start = time.perf_counter()
        detector.detect(frame)
        _sync_torch()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(timings_ms)
    median_ms = statistics.median(timings_ms)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    print(f"frames: {len(timings_ms)}")
    print(f"mean_ms: {mean_ms:.2f}")
    print(f"median_ms: {median_ms:.2f}")
    print(f"estimated_fps: {fps:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
