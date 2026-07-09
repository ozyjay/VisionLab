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

Use the consolidated PowerShell helper for normal setup:

```powershell
pwsh -File scripts/vision.ps1 setup -DownloadModels -ModelBundle demo
```

This creates the project-local ROCm virtual environment `.venv-rocm312`, installs
base and object-detection dependencies, installs the AMD ROCm PyTorch wheel set,
downloads the face detector, and downloads the practical demo model bundle:

- `yolo11s.pt`
- `yolo11m.pt`
- `yoloe-26s-seg.pt`
- `yoloe-26m-seg.pt`
- `mobileclip2_b.ts` for YOLOE text prompts

Use the larger bundle if you want more local model choices:

```powershell
pwsh -File scripts/vision.ps1 setup -DownloadModels -ModelBundle all-useful
```

The setup script refuses to install dependencies outside the project-local virtual
environment.
If you previously created the CPU `.venv`, it is no longer needed for the
recommended workflow once `.venv-rocm312` is working.

## Run

Start the preferred browser-rendered dashboard:

```powershell
pwsh -File scripts/vision.ps1 web
```

Then open <http://127.0.0.1:8019>. The browser dashboard is the preferred demo
UX; OpenCV remains the local capture and vision backend. The dashboard includes
a model selector for switching between downloaded `models/yolo*.pt` files and a
YOLOE prompt editor for changing the object labels during the demo.

Useful consolidated commands:

```powershell
pwsh -File scripts/vision.ps1 health
pwsh -File scripts/vision.ps1 config
pwsh -File scripts/vision.ps1 download-models -ModelBundle demo
pwsh -File scripts/vision.ps1 face-model
pwsh -File scripts/vision.ps1 benchmark -ModelPath models/yoloe-26s-seg.pt
pwsh -File scripts/vision.ps1 run     # legacy OpenCV window
```

YOLOE open-vocabulary prompts can be changed at launch:

```powershell
pwsh -File scripts/vision.ps1 web `
  -ModelPath models/yoloe-26m-seg.pt `
  -Prompts "person,mobile phone,computer mouse,pen,watch,keys,wallet,mug,keyboard,cable"
```

They can also be edited live from the browser dashboard sidebar.

YOLOE text prompts need Ultralytics' CLIP dependency and the
`models/mobileclip2_b.ts` text-encoder asset. The model downloader keeps this
asset in `models/` with the other local weights.

The older specialised scripts are still kept as compatibility shortcuts, but
`vision.ps1` is the recommended entrypoint.

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
| `VISION_WEB_HOST` | `127.0.0.1` | Browser dashboard host |
| `VISION_WEB_PORT` | `8019` | Browser dashboard port |

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
