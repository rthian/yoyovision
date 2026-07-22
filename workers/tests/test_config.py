"""Tests for worker `Settings` adapter env var parsing."""

from __future__ import annotations

from yoyovision_workers.config import Settings


def test_settings_default_adapters_are_mock(monkeypatch) -> None:
    monkeypatch.delenv("PIPELINE_POSE_ADAPTER", raising=False)
    settings = Settings()
    assert settings.pipeline_pose_adapter == "mock"
    assert settings.pipeline_hand_adapter == "mock"
    assert settings.pipeline_tracker_adapter == "mock"
    assert settings.pipeline_tracker_max_gap_ms == 500
    assert settings.pipeline_tracker_static_camera is False


def test_settings_reads_adapter_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_POSE_ADAPTER", "mediapipe")
    monkeypatch.setenv("PIPELINE_HAND_ADAPTER", "mediapipe")
    monkeypatch.setenv("PIPELINE_TRACKER_ADAPTER", "kalman")
    monkeypatch.setenv("PIPELINE_TRACKER_MAX_GAP_MS", "750")
    monkeypatch.setenv("PIPELINE_TRACKER_STATIC_CAMERA", "true")
    settings = Settings()
    assert settings.pipeline_pose_adapter == "mediapipe"
    assert settings.pipeline_hand_adapter == "mediapipe"
    assert settings.pipeline_tracker_adapter == "kalman"
    assert settings.pipeline_tracker_max_gap_ms == 750
    assert settings.pipeline_tracker_static_camera is True
