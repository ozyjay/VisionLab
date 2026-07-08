# Local AI Vision Assistant

Local-first computer-vision demo for a Framework Desktop with AMD/ROCm-friendly
fallbacks. The current implementation covers **MVP 0**, **MVP 1**, and
**MVP 2**:

- environment and camera health checks
- webcam viewer with FPS and frame-size overlay
- keyboard controls for capture, object detection, and placeholder modes
- optional Ultralytics YOLO object detection with CPU-first defaults
- cached object detections between inference frames for smoother display
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

To also try the optional Ultralytics YOLO object-detection backend:

```bash
bash scripts/setup.sh --object-detection
. .venv/bin/activate
```

PowerShell:

```powershell
pwsh -File scripts/setup.ps1
./.venv/bin/Activate.ps1
```

PowerShell with optional object detection:

```powershell
pwsh -File scripts/setup.ps1 -ObjectDetection
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

Run with object detection enabled:

```bash
VISION_ENABLE_OBJECT_DETECTION=true \
VISION_OBJECT_MODEL_PATH=models/yolo11n.pt \
python -m src.main run
```

The app will not download models automatically. Put a local YOLO model file at
`models/yolo11n.pt`, or point `VISION_OBJECT_MODEL_PATH` to another local model.
If the file or optional dependency is missing, the webcam app still starts and
prints a clear warning.

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
| `VISION_ENABLE_OBJECT_DETECTION` | `false` | Start with object detection enabled |
| `VISION_OBJECT_MODEL_PATH` | `models/yolo11n.pt` | Local YOLO model file |
| `VISION_OBJECT_DETECTOR_BACKEND` | `ultralytics` | Object backend: `ultralytics`, `auto`, or `none` |
| `VISION_OBJECT_CONFIDENCE_THRESHOLD` | `0.35` | Minimum detection confidence |
| `VISION_OBJECT_DETECTION_INTERVAL` | `3` | Run object inference every N frames |
| `VISION_OBJECT_DEVICE` | `cpu` | Ultralytics device, defaulting to CPU |
| `VISION_ENABLE_FACE_DETECTION` | `false` | Face overlay placeholder for MVP 1 |
| `VISION_ENABLE_VLLM` | `false` | Enable vLLM health check |
| `VISION_VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible vLLM base URL |
| `VISION_VLLM_MODEL` | `local-model` | Future vLLM model name |
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
| `o` | Toggle object detection on or off |
| `p` | Toggle privacy blur placeholder |
| `h` | Show or hide help |

## Privacy notes

- Face identity recognition is **not implemented**.
- Face embedding storage is **not implemented**.
- Saved frames are only written when you press `s`.
- Future face detection will support privacy blur without identifying people.

## Object detection notes

- The detector interface is `detect(frame) -> list[Detection]`.
- `Detection` contains `label`, `confidence`, `bbox`, and `source`.
- Missing model files do not crash the app.
- Ultralytics is optional so the base webcam demo remains lightweight.
- CPU is the default device. Do not set CUDA-specific options unless you have a
  matching local stack.

## Next MVP

Next step: **MVP 3 — Face detection and privacy mode**. Add lightweight face
detection, draw face boxes, and apply privacy blur without identity recognition
or face embedding storage.
