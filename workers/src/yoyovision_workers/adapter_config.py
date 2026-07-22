"""Resolves worker pipeline adapter names and kwargs from `Settings`.

Keeps `pipeline_runner.py` thin: register real adapters when needed, build
per-adapter constructor kwargs (e.g. Kalman gap bridging), and leave mock as
the safe default for dev/test environments without optional ML deps installed.
"""

from __future__ import annotations

from collections.abc import Mapping

from yoyovision_workers.config import Settings


def uses_non_mock_perception(settings: Settings) -> bool:
    """True when any perception-stage adapter is not the packaged mock."""
    return any(
        name != "mock"
        for name in (
            settings.pipeline_pose_adapter,
            settings.pipeline_hand_adapter,
            settings.pipeline_yoyo_adapter,
            settings.pipeline_tracker_adapter,
        )
    )


def ensure_real_adapters_registered(settings: Settings) -> None:
    """Import `yoyovision_ml.perception` so mediapipe/kalman/pytorch/onnx
    factories are registered before `run_analysis_pipeline` resolves names."""
    if uses_non_mock_perception(settings):
        import yoyovision_ml.perception  # noqa: F401


def build_pipeline_adapter_kwargs(settings: Settings) -> Mapping[str, Mapping[str, object]]:
    """Per-role constructor kwargs forwarded to `adapters_registry.create_*`."""
    kwargs: dict[str, dict[str, object]] = {}
    if settings.pipeline_tracker_adapter == "kalman":
        kwargs["tracker"] = {
            "max_gap_ms": settings.pipeline_tracker_max_gap_ms,
            "static_camera": settings.pipeline_tracker_static_camera,
        }
    return kwargs
