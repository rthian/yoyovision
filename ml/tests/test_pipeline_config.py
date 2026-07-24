from __future__ import annotations

from dataclasses import dataclass

from yoyovision_ml.pipeline_config import PipelineAdapterConfig, resolve_effective_pipeline_config


@dataclass
class _FakeSettings:
    pipeline_pose_adapter: str = "mock"
    pipeline_hand_adapter: str = "mock"
    pipeline_yoyo_adapter: str = "mock"
    pipeline_tracker_adapter: str = "mock"
    pipeline_temporal_event_adapter: str = "mock"
    pipeline_sample_fps: float = 15.0
    pipeline_device: str = "cpu"
    pipeline_tracker_max_gap_ms: int = 500
    pipeline_tracker_static_camera: bool = False
    pipeline_temporal_event_weights: str | None = None
    pipeline_yoyo_weights: str | None = None


def test_resolve_effective_pipeline_config_uses_settings_defaults() -> None:
    effective = resolve_effective_pipeline_config(None, _FakeSettings())
    assert effective.pose_adapter == "mock"
    assert effective.sample_fps == 15.0
    assert effective.adapter_kwargs == {}


def test_resolve_effective_pipeline_config_merges_nested_adapter_kwargs() -> None:
    settings = _FakeSettings(
        pipeline_tracker_adapter="kalman",
        pipeline_tracker_max_gap_ms=400,
    )
    job_config = PipelineAdapterConfig(
        adapter_kwargs={"tracker": {"static_camera": True}},
    )
    effective = resolve_effective_pipeline_config(job_config, settings)
    assert effective.adapter_kwargs["tracker"] == {
        "max_gap_ms": 400,
        "static_camera": True,
    }
