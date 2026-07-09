# Local AI Vision Assistant

Local-first computer-vision demo for a Framework Desktop with AMD/ROCm-friendly
fallbacks. The current implementation covers **MVP 0** through **MVP 4**:

- environment and camera health checks
- dashboard-style webcam viewer with FPS, frame size, and scene summary panels
- polished local OpenCV dashboard renderer with live demo commentary
- resizable viewer window
- keyboard controls for capture, object detection, and privacy modes
- optional Ultralytics YOLO object detection with CPU-first defaults
- ROCm-aware GPU mode helpers for Framework Desktop
- non-identifying OpenCV face detection and privacy blur
- cached object detections between inference frames for smoother display
- on-demand JSONL scene-state snapshots for objects, generic face counts, FPS,
  frame ID, timestamp, and privacy settings
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
logs/       # optional JSONL scene-state logs
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

### Try YOLOE open-vocabulary detection

For broader "random object" coverage, use a YOLOE 26-series model with explicit
text prompts. YOLOE is still local, but it can be prompted for classes outside
the fixed COCO label set:

```powershell
pwsh -File scripts/download-yolo-model.ps1 yoloe-s
pwsh -File scripts/run-yolo.ps1 `
  -ModelPath models/yoloe-26s-seg.pt `
  -Backend yoloe `
  -Prompts "person,phone,keys,wallet,remote control,mug,cable,mouse,keyboard,book,screwdriver"
```

If you already downloaded another YOLOE 26 model, point `-ModelPath` at it, for
example `models/yoloe-26m-seg.pt` or `models/yoloe-26l-seg.pt`. Use
`VISION_OBJECT_PROMPTS` to tune what the detector should look for in your room.
YOLOE text prompts also need Ultralytics' CLIP dependency and may download a
`mobileclip2_b.ts` text-encoder asset on first use.

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
| `VISION_CAMERA_RESOLUTION_MODE` | `fast` | Camera preset: `fast` = 640x480, `quality` = 1280x720 |
| `VISION_ENABLE_OBJECT_DETECTION` | `false` | Start with object detection enabled |
| `VISION_OBJECT_MODEL_PATH` | `models/yolo11n.pt` | Local YOLO model file |
| `VISION_OBJECT_DETECTOR_BACKEND` | `ultralytics` | Object backend: `ultralytics`, `yoloe`, `auto`, or `none` |
| `VISION_OBJECT_PROMPTS` | empty | Comma-separated YOLOE classes, for example `phone,keys,wallet` |
| `VISION_OBJECT_CONFIDENCE_THRESHOLD` | `0.35` | Minimum detection confidence |
| `VISION_OBJECT_DETECTION_INTERVAL` | `3` | Run object inference every N frames |
| `VISION_OBJECT_DETECTION_HOLD_FRAMES` | `8` | Keep previous boxes briefly when a detector frame returns empty, reducing demo flicker |
| `VISION_OBJECT_DEVICE` | `cpu` | Ultralytics/Torch device: `cpu`, `auto`, `rocm`, or `cuda:0` |
| `VISION_ENABLE_FACE_DETECTION` | `false` | Start with non-identifying face detection enabled |
| `VISION_ENABLE_VLLM` | `false` | Enable vLLM health check |
| `VISION_VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible vLLM base URL |
| `VISION_VLLM_MODEL` | `local-model` | Future vLLM model name |
| `VISION_FACE_MODEL_PATH` | `models/face_detection_yunet_2026may.onnx` | Local OpenCV YuNet face detector |
| `VISION_CAPTURES_DIR` | `captures` | Saved frame directory |
| `VISION_LOGS_DIR` | `logs` | Reserved log directory |
| `VISION_SCENE_STATE_INTERVAL_SECONDS` | `0` | Print JSONL scene-state snapshots every N seconds; `0` disables interval output |
| `VISION_SCENE_STATE_LOG_PATH` | empty | Optional JSONL file path to append emitted scene-state snapshots |

Example:

```bash
VISION_CAMERA_INDEX=1 VISION_TARGET_FPS=15 python -m src.main run
```

Use the quality camera preset when detection detail matters more than speed:

```powershell
pwsh -File scripts/run-gpu.ps1 -ResolutionMode quality
```

## Keyboard controls

| Key | Action |
| --- | --- |
| `q` | Quit |
| `s` | Save current displayed frame to `captures/` |
| `j` | Print one scene-state JSONL snapshot |
| `f` | Toggle non-identifying face detection |
| `o` | Toggle object detection on or off |
| `p` | Toggle privacy blur for detected faces |
| `h` | Show or hide help |

The right-hand panel includes concise commentary for demo audiences, including
which detector is active, what YOLOE prompts are being used, and which privacy
protections are in force.

## Scene-state JSONL

Press `j` in the viewer to print one compact JSON line to stdout. Set
`VISION_SCENE_STATE_INTERVAL_SECONDS` to a positive number to print snapshots
automatically at that cadence. Set `VISION_SCENE_STATE_LOG_PATH` to also append
emitted snapshots to a JSONL file.

Scene-state snapshots include counts and bounding boxes only. They do not
include image pixels, face crops, identities, embeddings, or cloud calls.

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
- Standard YOLO11 models are closed-set detectors. YOLOE is the preferred option
  when you need prompted detection of less common local objects.
- CPU is the default device. GPU mode is documented in `docs/GPU_MODE.md`.
- For ROCm PyTorch, `VISION_OBJECT_DEVICE=auto` resolves to `cuda:0` when the
  AMD GPU is available because PyTorch uses the `torch.cuda` namespace for ROCm.

## Next MVP

Next step: **MVP 5 — vLLM scene explanation**. Send compact scene state to a
local OpenAI-compatible vLLM endpoint with privacy-aware prompts.
