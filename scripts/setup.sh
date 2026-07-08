#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VISION_VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_OBJECT_DETECTION="${VISION_INSTALL_OBJECT_DETECTION:-0}"

for arg in "$@"; do
    case "$arg" in
        --object-detection)
            INSTALL_OBJECT_DETECTION="1"
            ;;
        --help|-h)
            echo "Usage: scripts/setup.sh [--object-detection]"
            echo
            echo "Options:"
            echo "  --object-detection  Install optional Ultralytics YOLO dependencies."
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: scripts/setup.sh [--object-detection]" >&2
            exit 2
            ;;
    esac
done

cd "$PROJECT_ROOT"

echo "Local AI Vision Assistant setup"
echo "Project: $PROJECT_ROOT"
echo "Virtual environment: $VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating project-local virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Error: expected Python executable was not found at $VENV_PYTHON" >&2
    exit 1
fi

ACTUAL_PREFIX="$("$VENV_PYTHON" -c 'import sys; print(sys.prefix)')"
EXPECTED_PREFIX="$(cd "$VENV_DIR" && pwd)"

if [[ "$ACTUAL_PREFIX" != "$EXPECTED_PREFIX" ]]; then
    echo "Error: refusing to install outside the project virtual environment." >&2
    echo "Expected prefix: $EXPECTED_PREFIX" >&2
    echo "Actual prefix:   $ACTUAL_PREFIX" >&2
    exit 1
fi

echo "Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip

echo "Installing dependencies..."
"$VENV_PYTHON" -m pip install -r requirements.txt

if [[ "$INSTALL_OBJECT_DETECTION" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    echo "Installing optional object-detection dependencies..."
    "$VENV_PYTHON" -m pip install -r requirements-object-detection.txt
else
    echo "Skipping optional object-detection dependencies."
    echo "Install them later with:"
    echo "  .venv/bin/python -m pip install -r requirements-object-detection.txt"
fi

echo
echo "Setup complete."
echo "Activate with:"
echo "  source .venv/bin/activate"
echo
echo "Run health check:"
echo "  .venv/bin/python -m src.main health"
