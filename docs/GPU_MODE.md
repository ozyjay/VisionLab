# GPU mode on Framework Desktop

This project targets the Framework Desktop with AMD Ryzen AI Max / Radeon 8060S
using ROCm through PyTorch.

## Local findings

On this machine, `rocminfo` reports:

- CPU/APU: `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`
- GPU agent: `AMD Radeon 8060S Graphics`
- GPU ISA: `gfx1151`
- Compute units: `40`
- Fast FP16: enabled

The recommended VisionLab environment on this machine is the Python 3.12 ROCm
environment named `.venv-rocm312`.

## Why Python 3.12 and `.venv-rocm312`?

AMD's current Ryzen Linux compatibility matrix lists:

- ROCm `7.2.1`
- PyTorch `2.9.1`
- Python `3.12`
- Ryzen AI Max+ 395 / `gfx1151`

The ROCm wheel set is large and Python-version-specific, so the project uses a
clearly named local environment rather than mixing CPU and ROCm Torch installs.

## Setup

```powershell
pwsh -File scripts/vision.ps1 setup
```

This installs:

- base app dependencies
- optional object-detection dependencies
- AMD ROCm PyTorch wheels from `repo.radeon.com`

The script refuses to install outside the selected virtual environment.

## Run browser dashboard with GPU auto-detection

```powershell
pwsh -File scripts/vision.ps1 web
```

Defaults:

- model: `models/yoloe-26s-seg.pt`
- device: `auto`
- camera resolution mode: `quality` (`1920x1080`, or 1080p)
- object detection: enabled

To run a bigger model:

```powershell
pwsh -File scripts/vision.ps1 web -ModelPath models/yoloe-26m-seg.pt
```

To run a YOLOE 26-series open-vocabulary model:

```powershell
pwsh -File scripts/vision.ps1 web `
  -ModelPath models/yoloe-26s-seg.pt `
  -Prompts "person,phone,keys,wallet,remote control,mug,cable"
```

## Benchmark YOLO

```powershell
pwsh -File scripts/vision.ps1 benchmark
pwsh -File scripts/vision.ps1 benchmark -ModelPath models/yolo11m.pt -Frames 60
pwsh -File scripts/vision.ps1 benchmark -ModelPath models/yoloe-26s-seg.pt -Prompts "phone,keys,wallet"
```

## Device behaviour

`VISION_OBJECT_DEVICE=auto` resolves to:

- `cuda:0` when Torch reports an available GPU
- `cpu` otherwise

This is expected for ROCm: PyTorch uses the `torch.cuda` API namespace for AMD
GPUs too. Check `torch.version.hip` to confirm that the Torch build is ROCm/HIP.

## Sources

- AMD Ryzen Linux compatibility matrix:
  <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html>
- AMD Ryzen PyTorch install guide:
  <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-pytorch.html>
