"""Helpers for resolving local Torch accelerator devices."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any


AUTO_DEVICE_ALIASES = {"auto", "gpu", "accelerator"}
ROCM_DEVICE_ALIASES = {"rocm", "hip"}


@dataclass(slots=True)
class TorchAcceleratorStatus:
    """Torch accelerator details for health checks and detector setup."""

    requested_device: str
    resolved_device: str
    torch_available: bool
    torch_version: str | None = None
    hip_version: str | None = None
    cuda_version: str | None = None
    gpu_available: bool = False
    device_count: int = 0
    device_name: str | None = None
    backend: str = "cpu"
    note: str = ""

    @property
    def is_rocm(self) -> bool:
        """Return whether the active Torch build reports ROCm/HIP support."""

        return bool(self.hip_version)


def _normalise_device(value: str | None) -> str:
    """Return a compact requested device value."""

    return (value or "cpu").strip().lower() or "cpu"


def _torch_module_available() -> bool:
    """Return whether Torch can be imported without importing it yet."""

    return importlib.util.find_spec("torch") is not None


def get_torch_accelerator_status(requested_device: str | None = "cpu") -> TorchAcceleratorStatus:
    """Resolve a requested Torch device against the local environment.

    PyTorch uses the ``cuda`` namespace for both NVIDIA CUDA and AMD ROCm/HIP
    backends. For ROCm, a resolved device such as ``cuda:0`` still means the AMD
    GPU when ``torch.version.hip`` is populated.
    """

    requested = _normalise_device(requested_device)

    if requested == "cpu":
        return TorchAcceleratorStatus(
            requested_device=requested,
            resolved_device="cpu",
            torch_available=_torch_module_available(),
            backend="cpu",
            note="CPU requested.",
        )

    if not _torch_module_available():
        fallback_note = "Torch is not installed; falling back to CPU."
        if requested not in AUTO_DEVICE_ALIASES | ROCM_DEVICE_ALIASES:
            fallback_note = "Torch is not installed; explicit accelerator device may fail."
            return TorchAcceleratorStatus(
                requested_device=requested,
                resolved_device=requested,
                torch_available=False,
                backend="unknown",
                note=fallback_note,
            )

        return TorchAcceleratorStatus(
            requested_device=requested,
            resolved_device="cpu",
            torch_available=False,
            backend="cpu",
            note=fallback_note,
        )

    try:
        import torch

        torch_version = str(getattr(torch, "__version__", "unknown"))
        version_info: Any = getattr(torch, "version", None)
        hip_version = getattr(version_info, "hip", None)
        cuda_version = getattr(version_info, "cuda", None)
        gpu_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
        device_name = torch.cuda.get_device_name(0) if gpu_available else None
    except Exception as exc:
        return TorchAcceleratorStatus(
            requested_device=requested,
            resolved_device="cpu" if requested in AUTO_DEVICE_ALIASES | ROCM_DEVICE_ALIASES else requested,
            torch_available=True,
            backend="unknown",
            note=f"Torch accelerator check failed: {exc}",
        )

    backend = "rocm" if hip_version else ("cuda" if cuda_version else "cpu")

    if requested in AUTO_DEVICE_ALIASES:
        if gpu_available:
            return TorchAcceleratorStatus(
                requested_device=requested,
                resolved_device="cuda:0",
                torch_available=True,
                torch_version=torch_version,
                hip_version=hip_version,
                cuda_version=cuda_version,
                gpu_available=gpu_available,
                device_count=device_count,
                device_name=device_name,
                backend=backend,
                note=f"Auto-selected {backend} GPU.",
            )

        return TorchAcceleratorStatus(
            requested_device=requested,
            resolved_device="cpu",
            torch_available=True,
            torch_version=torch_version,
            hip_version=hip_version,
            cuda_version=cuda_version,
            gpu_available=gpu_available,
            device_count=device_count,
            device_name=device_name,
            backend="cpu",
            note="No Torch GPU is available; falling back to CPU.",
        )

    if requested in ROCM_DEVICE_ALIASES:
        if gpu_available and hip_version:
            return TorchAcceleratorStatus(
                requested_device=requested,
                resolved_device="cuda:0",
                torch_available=True,
                torch_version=torch_version,
                hip_version=hip_version,
                cuda_version=cuda_version,
                gpu_available=gpu_available,
                device_count=device_count,
                device_name=device_name,
                backend="rocm",
                note="ROCm/HIP GPU selected.",
            )

        return TorchAcceleratorStatus(
            requested_device=requested,
            resolved_device="cpu",
            torch_available=True,
            torch_version=torch_version,
            hip_version=hip_version,
            cuda_version=cuda_version,
            gpu_available=gpu_available,
            device_count=device_count,
            device_name=device_name,
            backend="cpu",
            note="ROCm/HIP was requested but is not available; falling back to CPU.",
        )

    return TorchAcceleratorStatus(
        requested_device=requested,
        resolved_device=requested,
        torch_available=True,
        torch_version=torch_version,
        hip_version=hip_version,
        cuda_version=cuda_version,
        gpu_available=gpu_available,
        device_count=device_count,
        device_name=device_name,
        backend=backend,
        note="Explicit Torch device requested.",
    )
