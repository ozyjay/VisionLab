# Local AI Vision Assistant

Local-first computer-vision demo for a Framework Desktop with AMD/ROCm-friendly
fallbacks. The first implementation covers **MVP 0** and **MVP 1**:

- environment and camera health checks
- webcam viewer with FPS and frame-size overlay
- keyboard controls for capture and placeholder modes
- resilient behaviour when optional GPU/model dependencies are unavailable

The app does **not** identify people, store face data, or use cloud services.

## Project structure

```text
src/
  main.py
  camera.py
  config.py
  detectors/
  scene_state.py
  vllm_client.py
  ui/
docs/
scripts/
captures/   # created when saving frames
logs/       # reserved for later JSONL scene-state logging
```

## Install

Use a project-local virtual environment:

```bash
bash scripts/setup.sh
. .venv/bin/activate
```

PowerShell:

```powershell
pwsh -File scripts/setup.ps1
./.venv/bin/Activate.ps1
```

The setup scripts refuse to install dependencies outside the project-local
virtual environment.

## Run

Health check:

```bash
python -m src.main health
```

Open the webcam viewer:

```bash
python -m src.main run
```

If no command is supplied, the app defaults to `run`.

Convenience scripts are also available:

```bash
bash scripts/health.sh
bash scripts/run.sh
```

PowerShell:

```powershell
pwsh -File scripts/health.ps1
pwsh -File scripts/run.ps1
```

## Configuration

Configuration is loaded from environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VISION_CAMERA_INDEX` | `0` | OpenCV camera index |
| `VISION_TARGET_FPS` | `30` | Target display FPS |
| `VISION_ENABLE_OBJECT_DETECTION` | `false` | Object overlay placeholder for MVP 1 |
| `VISION_ENABLE_FACE_DETECTION` | `false` | Face overlay placeholder for MVP 1 |
| `VISION_ENABLE_VLLM` | `false` | Enable vLLM health check |
| `VISION_VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible vLLM base URL |
| `VISION_VLLM_MODEL` | `local-model` | Future vLLM model name |
| `VISION_OBJECT_MODEL_PATH` | `models/object_detector.onnx` | Future object detector model |
| `VISION_FACE_MODEL_PATH` | `models/face_detector.onnx` | Future face detector model |
| `VISION_CAPTURES_DIR` | `captures` | Saved frame directory |
| `VISION_LOGS_DIR` | `logs` | Future scene-state log directory |

Example:

```bash
VISION_CAMERA_INDEX=1 VISION_TARGET_FPS=15 python -m src.main run
```

## Keyboard controls

| Key | Action |
| --- | --- |
| `q` | Quit |
| `s` | Save current displayed frame to `captures/` |
| `f` | Toggle face overlay placeholder |
| `o` | Toggle object overlay placeholder |
| `p` | Toggle privacy blur placeholder |
| `h` | Show or hide help |

## Privacy notes

- Face identity recognition is **not implemented**.
- Face embedding storage is **not implemented**.
- Saved frames are only written when you press `s`.
- Future face detection will support privacy blur without identifying people.

## Next MVP

Next step: **MVP 2 — Object detection**. Add a real `ObjectDetector` backend
using a small YOLO model first, with CPU/ONNX Runtime fallback behaviour and
cached detections between inference frames.
