"""Download an Ultralytics YOLO model into the local models directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys


YOLO11_ALIASES = {
    "n": "yolo11n.pt",
    "nano": "yolo11n.pt",
    "s": "yolo11s.pt",
    "small": "yolo11s.pt",
    "m": "yolo11m.pt",
    "medium": "yolo11m.pt",
    "l": "yolo11l.pt",
    "large": "yolo11l.pt",
    "x": "yolo11x.pt",
    "extra-large": "yolo11x.pt",
}

YOLOE_ALIASES = {
    "yoloe": "yoloe-26s-seg.pt",
    "yoloe-s": "yoloe-26s-seg.pt",
    "yoloe-small": "yoloe-26s-seg.pt",
    "yoloe-m": "yoloe-26m-seg.pt",
    "yoloe-medium": "yoloe-26m-seg.pt",
    "yoloe-l": "yoloe-26l-seg.pt",
    "yoloe-large": "yoloe-26l-seg.pt",
    "yoloe-x": "yoloe-26x-seg.pt",
    "yoloe-pf": "yoloe-26s-seg-pf.pt",
    "yoloe-prompt-free": "yoloe-26s-seg-pf.pt",
}

MODEL_NAME_PATTERN = re.compile(r"^yolo[\w.-]*\.pt$")


def resolve_model_name(value: str) -> str:
    """Resolve a short model alias or validate an Ultralytics model filename."""

    name = value.strip()
    if not name:
        raise ValueError("model name cannot be empty")

    alias = YOLO11_ALIASES.get(name.lower())
    if alias:
        return alias

    alias = YOLOE_ALIASES.get(name.lower())
    if alias:
        return alias

    if "/" in name or "\\" in name:
        raise ValueError("use a model filename such as yolo11m.pt, not a path")

    if not MODEL_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "expected a YOLO .pt filename such as yolo11s.pt, yolo11m.pt, "
            "yolo11x.pt, or yoloe-26s-seg.pt"
        )

    return name


def download_model(model_name: str, models_dir: Path) -> Path:
    """Download the requested model through Ultralytics and return its local path."""

    target = models_dir / model_name
    if target.exists():
        print(f"Model already exists: {target}")
        return target

    try:
        import ultralytics
    except ImportError:
        print(
            "Ultralytics is not installed. Install optional object-detection dependencies first:",
            file=sys.stderr,
        )
        print(
            "  .venv/bin/python -m pip install -r requirements-object-detection.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)

    model_class_name = "YOLOE" if model_name.startswith("yoloe") else "YOLO"
    model_class = getattr(ultralytics, model_class_name, None)
    if model_class is None:
        print(
            f"Ultralytics does not provide {model_class_name}. Upgrade it first:",
            file=sys.stderr,
        )
        print("  .venv/bin/python -m pip install --upgrade ultralytics", file=sys.stderr)
        raise SystemExit(1)

    models_dir.mkdir(parents=True, exist_ok=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(models_dir)
        print(f"Downloading/loading {model_name} with Ultralytics...")
        model_class(model_name)
    finally:
        os.chdir(previous_cwd)

    if not target.exists():
        raise RuntimeError(
            f"Ultralytics completed but {target} was not found. "
            "Check the model name and Ultralytics output above."
        )

    print(f"Model ready: {target}")
    return target


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Download a YOLO model to ./models for the local vision demo."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="yolo11s.pt",
        help=(
            "Model filename or YOLO11 size alias. Examples: small, medium, large, "
            "x, yolo11m.pt, yoloe-s, yoloe-26s-seg.pt. Default: yolo11s.pt"
        ),
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Destination directory for downloaded weights. Default: models",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the downloader."""

    args = build_parser().parse_args(argv)
    try:
        model_name = resolve_model_name(args.model)
        download_model(model_name, Path(args.models_dir))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
