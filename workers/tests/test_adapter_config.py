"""Tests for adapter kwargs resolution from worker settings."""

from __future__ import annotations

from yoyovision_workers.adapter_config import (
    build_pipeline_adapter_kwargs,
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


def test_uses_non_mock_perception_detects_mediapipe() -> None:
    settings = Settings(pipeline_pose_adapter="mediapipe")
    assert uses_non_mock_perception(settings) is True


def test_uses_non_mock_perception_false_for_all_mock() -> None:
    settings = Settings()
    assert uses_non_mock_perception(settings) is False


def test_build_pipeline_adapter_kwargs_includes_torch_weights() -> None:
    settings = Settings(
        pipeline_temporal_event_adapter="torch",
        pipeline_temporal_event_weights="/models/events.pt",
    )
    assert build_pipeline_adapter_kwargs(settings) == {
        "temporal_event": {"weights_path": "/models/events.pt"}
    }


def test_uses_non_mock_temporal_for_torch_adapter() -> None:
    settings = Settings(pipeline_temporal_event_adapter="torch")
    assert uses_non_mock_temporal(settings) is True
