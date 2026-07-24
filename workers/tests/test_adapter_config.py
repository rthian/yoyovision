"""Tests for adapter kwargs resolution from worker settings."""

from __future__ import annotations

from yoyovision_ml.pipeline_config import PipelineAdapterConfig
from yoyovision_workers.adapter_config import (
    build_pipeline_adapter_kwargs,
    resolve_job_pipeline_config,
    uses_non_mock_perception,
    uses_non_mock_temporal,
)
from yoyovision_workers.config import Settings


def test_build_pipeline_adapter_kwargs_empty_for_mock_tracker() -> None:
    settings = Settings(pipeline_tracker_adapter="mock")
    assert build_pipeline_adapter_kwargs(settings) == {}


def test_build_pipeline_adapter_kwargs_includes_kalman_tuning() -> None:
    settings = Settings(
        pipeline_tracker_adapter="kalman",
        pipeline_tracker_max_gap_ms=900,
        pipeline_tracker_static_camera=True,
    )
    assert build_pipeline_adapter_kwargs(settings) == {
        "tracker": {"max_gap_ms": 900, "static_camera": True}
    }


def test_build_pipeline_adapter_kwargs_includes_yoyo_pytorch_weights() -> None:
    settings = Settings(
        pipeline_yoyo_adapter="pytorch",
        pipeline_yoyo_weights="/models/yoyo.pt",
    )
    assert build_pipeline_adapter_kwargs(settings) == {
        "yoyo": {"weights_path": "/models/yoyo.pt"}
    }


def test_build_pipeline_adapter_kwargs_includes_torch_weights() -> None:
    settings = Settings(
        pipeline_temporal_event_adapter="torch",
        pipeline_temporal_event_weights="/models/events.pt",
    )
    assert build_pipeline_adapter_kwargs(settings) == {
        "temporal_event": {"weights_path": "/models/events.pt"}
    }


def test_resolve_job_pipeline_config_merges_job_overrides() -> None:
    settings = Settings(
        pipeline_pose_adapter="mock",
        pipeline_temporal_event_adapter="mock",
        pipeline_sample_fps=15.0,
        pipeline_device="cpu",
    )
    job_config = PipelineAdapterConfig(
        pose_adapter="mediapipe",
        temporal_event_adapter="torch",
        sample_fps=20.0,
        adapter_kwargs={"temporal_event": {"weights_path": "/job/events.pt"}},
    )
    effective = resolve_job_pipeline_config(job_config, settings)
    assert effective.pose_adapter == "mediapipe"
    assert effective.temporal_event_adapter == "torch"
    assert effective.sample_fps == 20.0
    assert effective.adapter_kwargs["temporal_event"] == {"weights_path": "/job/events.pt"}


def test_uses_non_mock_perception_detects_mediapipe() -> None:
    settings = Settings(pipeline_pose_adapter="mediapipe")
    effective = resolve_job_pipeline_config(None, settings)
    assert uses_non_mock_perception(effective) is True


def test_uses_non_mock_perception_false_for_all_mock() -> None:
    settings = Settings()
    effective = resolve_job_pipeline_config(None, settings)
    assert uses_non_mock_perception(effective) is False


def test_uses_non_mock_temporal_for_torch_adapter() -> None:
    settings = Settings(pipeline_temporal_event_adapter="torch")
    effective = resolve_job_pipeline_config(None, settings)
    assert uses_non_mock_temporal(effective) is True
