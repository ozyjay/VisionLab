# Local AI Vision Assistant MVP Plan

This project is intentionally staged so each milestone can run independently
before adding the next layer.

## MVP 0 — Project scaffold and health check

- Create a clean Python project structure.
- Load configuration from environment variables.
- Provide a health check for Python, OpenCV, camera access, torch, ROCm/HIP,
  ONNX Runtime, and the optional local vLLM endpoint.
- Never hard-fail just because GPU acceleration or optional packages are
  unavailable.

## MVP 1 — Webcam viewer

- Open a local webcam with OpenCV.
- Show FPS, frame size, and active modes.
- Support keyboard controls for quitting, saving frames, and toggling
  face detection, object detection, and privacy blur.

## MVP 2 — Object detection

- Add a swappable `ObjectDetector` implementation.
- Prefer a small YOLO model first, with CPU and ONNX Runtime fallback paths.
- Cache detections between inference frames to keep the UI smooth.

## MVP 3 — Face detection and privacy mode

- Add lightweight face detection without identity recognition.
- Blur detected face regions when privacy mode is enabled.
- Do not store faces, embeddings, or biometric identifiers.

## MVP 4 — Scene state JSON

- Combine detections into a structured scene-state object.
- Include counts, detections, FPS, timestamp, frame ID, and privacy settings.
- Print or log JSONL snapshots on demand or at a configured interval.
- Status: implemented with `j` for on-demand stdout output, optional interval
  output via `VISION_SCENE_STATE_INTERVAL_SECONDS`, and optional file append via
  `VISION_SCENE_STATE_LOG_PATH`.

## MVP 5 — vLLM scene explanation

- Send compact scene state to a local OpenAI-compatible vLLM endpoint.
- Keep prompts privacy-aware: describe visible objects, count faces, and state
  that identity recognition is disabled.

## MVP 6 — Basic UI polish

- Improve overlays and demo help.
- Add a simple text panel if useful.
- Keep documentation practical and clear for live demos.
