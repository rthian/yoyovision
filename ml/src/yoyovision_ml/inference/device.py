"""CPU/GPU device selection and runtime-version reporting.

Prompt F: "Support CPU and GPU execution." / "Record device, runtime and
model version." No GPU hardware is assumed to exist wherever this runs (this
repo's dev/demo environment is CPU-only) -- `resolve_device` always falls
back to CPU cleanly and explains why, rather than crashing when CUDA is
unavailable.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

_VALID_PREFERENCES = frozenset({"auto", "cpu", "cuda", "gpu"})


@dataclass(slots=True, frozen=True)
class DeviceInfo:
    """Result of resolving a preferred device against what is actually
    available on this worker process."""

    requested: str
    resolved: str
    available: bool
    reason: str


def resolve_device(preferred: str = "auto") -> DeviceInfo:
    """Resolves `preferred` ("auto" | "cpu" | "cuda" | "gpu") to an actually
    usable device string, falling back to `"cpu"` whenever GPU support was
    requested but is not available (missing `torch`, or no CUDA device).
    Never raises: an unresolvable GPU preference is a fallback, not a
    deterministic failure, since CPU is always a safe (if slower) fallback.
    """
    preferred_normalized = preferred.strip().lower() if preferred else "auto"
    if preferred_normalized not in _VALID_PREFERENCES:
        return DeviceInfo(
            requested=preferred,
            resolved="cpu",
            available=True,
            reason=f"Unknown device preference '{preferred}'; falling back to cpu.",
        )

    if preferred_normalized == "cpu":
        return DeviceInfo(requested=preferred, resolved="cpu", available=True, reason="cpu requested")

    try:
        import torch
    except ImportError:
        return DeviceInfo(
            requested=preferred,
            resolved="cpu",
            available=False,
            reason="torch is not installed; GPU execution requires the optional torch dependency.",
        )

    if torch.cuda.is_available():
        return DeviceInfo(
            requested=preferred,
            resolved="cuda",
            available=True,
            reason=f"CUDA available ({torch.cuda.get_device_name(0)}).",
        )

    return DeviceInfo(
        requested=preferred,
        resolved="cpu",
        available=False,
        reason="torch is installed but no CUDA device is available on this worker.",
    )


def runtime_versions() -> dict[str, str]:
    """Best-effort collection of interpreter/runtime versions relevant to
    reproducibility ("Record ... runtime and model version"). Optional
    dependencies that are not installed are simply omitted, never guessed."""
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import onnxruntime

        versions["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    try:
        import mediapipe

        versions["mediapipe"] = mediapipe.__version__
    except ImportError:
        pass
    return versions
