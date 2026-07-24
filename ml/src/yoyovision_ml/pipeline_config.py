"""Shared pipeline adapter configuration for API job rows and worker runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field


class PipelineSettingsLike(Protocol):
    pipeline_pose_adapter: str
    pipeline_hand_adapter: str
    pipeline_yoyo_adapter: str
    pipeline_tracker_adapter: str
    pipeline_temporal_event_adapter: str
    pipeline_sample_fps: float
    pipeline_device: str
    pipeline_tracker_max_gap_ms: int
    pipeline_tracker_static_camera: bool
    pipeline_temporal_event_weights: str | None
    pipeline_yoyo_weights: str | None


class PipelineAdapterConfig(BaseModel):
    """Optional per-job overrides merged over worker `Settings` at run time."""

    model_config = ConfigDict(extra="forbid")

    pose_adapter: str | None = None
    hand_adapter: str | None = None
    yoyo_adapter: str | None = None
    tracker_adapter: str | None = None
    temporal_event_adapter: str | None = None
    sample_fps: float | None = Field(default=None, gt=0)
    device: str | None = None
    adapter_kwargs: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True, slots=True)
class EffectivePipelineConfig:
    pose_adapter: str
    hand_adapter: str
    yoyo_adapter: str
    tracker_adapter: str
    temporal_event_adapter: str
    sample_fps: float
    device: str
    adapter_kwargs: dict[str, dict[str, object]]


def _deep_merge_kwargs(
    base: Mapping[str, Mapping[str, object]],
    override: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, object]]:
    if not override:
        return {role: dict(kwargs) for role, kwargs in base.items()}
    merged: dict[str, dict[str, object]] = {role: dict(kwargs) for role, kwargs in base.items()}
    for role, kwargs in override.items():
        merged.setdefault(role, {})
        merged[role].update(dict(kwargs))
    return merged


def build_default_adapter_kwargs(
    *,
    yoyo_adapter: str,
    tracker_adapter: str,
    temporal_event_adapter: str,
    tracker_max_gap_ms: int,
    tracker_static_camera: bool,
    temporal_event_weights: str | None,
    yoyo_weights: str | None,
) -> dict[str, dict[str, object]]:
    kwargs: dict[str, dict[str, object]] = {}
    if tracker_adapter == "kalman":
        kwargs["tracker"] = {
            "max_gap_ms": tracker_max_gap_ms,
            "static_camera": tracker_static_camera,
        }
    if temporal_event_adapter == "torch" and temporal_event_weights:
        kwargs["temporal_event"] = {"weights_path": temporal_event_weights}
    if yoyo_adapter == "pytorch" and yoyo_weights:
        kwargs["yoyo"] = {"weights_path": yoyo_weights}
    elif yoyo_adapter == "onnx" and yoyo_weights:
        kwargs["yoyo"] = {"model_path": yoyo_weights}
    return kwargs


def resolve_effective_pipeline_config(
    job_config: PipelineAdapterConfig | Mapping[str, object] | None,
    settings: PipelineSettingsLike,
) -> EffectivePipelineConfig:
    """Merges optional per-job overrides with worker-global defaults."""
    parsed: PipelineAdapterConfig | None
    if job_config is None:
        parsed = None
    elif isinstance(job_config, PipelineAdapterConfig):
        parsed = job_config
    else:
        parsed = PipelineAdapterConfig.model_validate(dict(job_config))

    pose_adapter = (parsed.pose_adapter if parsed else None) or settings.pipeline_pose_adapter
    hand_adapter = (parsed.hand_adapter if parsed else None) or settings.pipeline_hand_adapter
    yoyo_adapter = (parsed.yoyo_adapter if parsed else None) or settings.pipeline_yoyo_adapter
    tracker_adapter = (
        (parsed.tracker_adapter if parsed else None) or settings.pipeline_tracker_adapter
    )
    temporal_event_adapter = (
        (parsed.temporal_event_adapter if parsed else None)
        or settings.pipeline_temporal_event_adapter
    )
    sample_fps = (parsed.sample_fps if parsed else None) or settings.pipeline_sample_fps
    device = (parsed.device if parsed else None) or settings.pipeline_device

    base_kwargs = build_default_adapter_kwargs(
        yoyo_adapter=yoyo_adapter,
        tracker_adapter=tracker_adapter,
        temporal_event_adapter=temporal_event_adapter,
        tracker_max_gap_ms=settings.pipeline_tracker_max_gap_ms,
        tracker_static_camera=settings.pipeline_tracker_static_camera,
        temporal_event_weights=settings.pipeline_temporal_event_weights,
        yoyo_weights=settings.pipeline_yoyo_weights,
    )
    adapter_kwargs = _deep_merge_kwargs(
        base_kwargs,
        parsed.adapter_kwargs if parsed else None,
    )

    return EffectivePipelineConfig(
        pose_adapter=pose_adapter,
        hand_adapter=hand_adapter,
        yoyo_adapter=yoyo_adapter,
        tracker_adapter=tracker_adapter,
        temporal_event_adapter=temporal_event_adapter,
        sample_fps=sample_fps,
        device=device,
        adapter_kwargs=adapter_kwargs,
    )
