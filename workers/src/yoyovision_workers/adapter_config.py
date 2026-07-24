"""Resolves worker pipeline adapter names and kwargs from `Settings`.

Keeps `pipeline_runner.py` thin: register real adapters when needed, build
per-adapter constructor kwargs (e.g. Kalman gap bridging), and leave mock as
the safe default for dev/test environments without optional ML deps installed.
"""

from __future__ import annotations

from collections.abc import Mapping

from yoyovision_ml.pipeline_config import (
    EffectivePipelineConfig,
    PipelineAdapterConfig,
    resolve_effective_pipeline_config,
)

from yoyovision_workers.config import Settings


def uses_non_mock_perception(effective: EffectivePipelineConfig) -> bool:
    """True when any perception-stage adapter is not the packaged mock."""
    return any(
        name != "mock"
        for name in (
            effective.pose_adapter,
            effective.hand_adapter,
            effective.yoyo_adapter,
            effective.tracker_adapter,
        )
    )


def uses_non_mock_temporal(effective: EffectivePipelineConfig) -> bool:
    return effective.temporal_event_adapter != "mock"


def ensure_real_adapters_registered(effective: EffectivePipelineConfig) -> None:
    """Register real adapter factories before `run_analysis_pipeline` resolves names."""
    if uses_non_mock_perception(effective):
        import yoyovision_ml.perception  # noqa: F401
    if uses_non_mock_temporal(effective):
        import yoyovision_ml.events  # noqa: F401


def build_pipeline_adapter_kwargs(settings: Settings) -> Mapping[str, Mapping[str, object]]:
    """Per-role constructor kwargs from worker settings only (no job overrides)."""
    return resolve_effective_pipeline_config(None, settings).adapter_kwargs


def resolve_job_pipeline_config(
    job_config: PipelineAdapterConfig | Mapping[str, object] | None,
    settings: Settings,
) -> EffectivePipelineConfig:
    return resolve_effective_pipeline_config(job_config, settings)
