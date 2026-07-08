# Local AI Vision Assistant

Local-first computer-vision demo for a Framework Desktop with AMD/ROCm-friendly
fallbacks. The current implementation covers **MVP 0**, **MVP 1**, and
**MVP 2**:

- environment and camera health checks
- webcam viewer with FPS and frame-size overlay
- keyboard controls for capture, object detection, and privacy modes
- optional Ultralytics YOLO object detection with CPU-first defaults
- non-identifying OpenCV face detection and privacy blur
- cached object detections between inference frames for smoother display
- resilient behaviour when optional GPU/model dependencies are unavailable

The app does **not** identify people, compute or store face embeddings, store
face data, or use cloud services.

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

The app will not download models automatically at viewer startup. Put a local
YOLO model file at `models/yolo11n.pt`, or point `VISION_OBJECT_MODEL_PATH` to
another local model.
If the file or optional dependency is missing, the webcam app still starts and
prints a clear warning.

### Try stronger YOLO models

The bundled default is `yolo11n.pt` because it is small and CPU-friendly. To try
stronger Ultralytics YOLO11 detection models, download one explicitly with
PowerShell, then run the viewer against that local file:

```powershell
pwsh -File scripts/download-yolo-model.ps1 small
pwsh -File scripts/run-yolo.ps1 -ModelPath models/yolo11s.pt
```

Useful size aliases:

| Alias | Model | Notes |
| --- | --- | --- |
| `small` | `yolo11s.pt` | First upgrade to try; usually still practical on CPU |
| `medium` | `yolo11m.pt` | Better accuracy, noticeably heavier |
| `large` | `yolo11l.pt` | Stronger again; expect lower FPS |
| `x` | `yolo11x.pt` | Heaviest YOLO11 option; best for GPU testing |

The downloader also accepts explicit filenames such as `yolo11m.pt`. Ultralytics
documents YOLO11 detection filenames as `yolo11n.pt`, `yolo11s.pt`,
`yolo11m.pt`, `yolo11l.pt`, and `yolo11x.pt`:
<https://docs.ultralytics.com/models/yolo11/>.

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
| `VISION_FACE_MODEL_PATH` | `models/face_detection_yunet_2026may.onnx` | Local OpenCV YuNet face detector |
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
| `f` | Toggle non-identifying face detection |
| `o` | Toggle object detection on or off |
| `p` | Toggle privacy blur for detected faces |
| `h` | Show or hide help |

## Privacy notes

- Face identity recognition is **not implemented**.
- Face embeddings are **not implemented or stored**.
- Face detection only draws generic boxes and counts visible faces.
- Saved frames are only written when you press `s`.
- Privacy blur runs locally on detected face regions without identifying people.

To enable face detection, download OpenCV Zoo's YuNet detector:

```powershell
pwsh -File scripts/download-face-detector.ps1
```

## Object detection notes

- The detector interface is `detect(frame) -> list[Detection]`.
- `Detection` contains `label`, `confidence`, `bbox`, and `source`.
- Missing model files do not crash the app.
- Ultralytics is optional so the base webcam demo remains lightweight.
- CPU is the default device. Do not set CUDA-specific options unless you have a
  matching local stack.

## Next MVP

Next step: **MVP 4 — Scene-state summaries**. Summarise visible objects and
generic face counts without identity recognition or face embedding storage.
