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

The default `.venv` currently uses a non-ROCm Torch build, so GPU mode uses a
separate Python 3.12 environment named `.venv-rocm312`.

## Why a separate environment?

AMD's current Ryzen Linux compatibility matrix lists:

- ROCm `7.2.1`
- PyTorch `2.9.1`
- Python `3.12`
- Ryzen AI Max+ 395 / `gfx1151`

The project default `.venv` can stay CPU-stable while `.venv-rocm312` carries the
larger ROCm wheel set.

## Setup

```powershell
pwsh -File scripts/setup-rocm.ps1
```

This installs:

- base app dependencies
- optional object-detection dependencies
- AMD ROCm PyTorch wheels from `repo.radeon.com`

The script refuses to install outside the selected virtual environment.

## Run GPU viewer

```powershell
pwsh -File scripts/run-gpu.ps1
```

Defaults:

- model: `models/yolo11s.pt`
- device: `auto`
- camera resolution mode: `fast` (`640x480`)
- object detection: enabled

To run a bigger model:

```powershell
pwsh -File scripts/run-gpu.ps1 -ModelPath models/yolo11m.pt
```

To trade speed for more detection detail:

```powershell
pwsh -File scripts/run-gpu.ps1 -ResolutionMode quality
```

To include local non-identifying face detection:

```powershell
pwsh -File scripts/run-gpu.ps1 -FaceDetection
```

## Benchmark YOLO

```powershell
pwsh -File scripts/benchmark-yolo.ps1
pwsh -File scripts/benchmark-yolo.ps1 -ModelPath models/yolo11m.pt -Frames 60
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
