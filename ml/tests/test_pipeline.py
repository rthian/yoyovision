"""Tests for the end-to-end mock pipeline orchestrator (`yoyovision_ml.pipeline`)."""

from __future__ import annotations

from pathlib import Path

from yoyovision_ml.pipeline import run_analysis_pipeline
from yoyovision_ml.ruleset import default_ruleset


def test_run_analysis_pipeline_is_deterministic() -> None:
    video_path = Path("/tmp/does-not-need-to-exist-for-mock-adapters.mp4")
    ruleset = default_ruleset()

    result_a = run_analysis_pipeline(video_path, duration_ms=20_000, fps=30.0, ruleset=ruleset)
    result_b = run_analysis_pipeline(video_path, duration_ms=20_000, fps=30.0, ruleset=ruleset)

    assert result_a.events == result_b.events
    assert result_a.deductions == result_b.deductions
    assert result_a.score == result_b.score


def test_run_analysis_pipeline_produces_labelled_mock_events_and_score() -> None:
    video_path = Path("/tmp/does-not-need-to-exist-for-mock-adapters.mp4")
    ruleset = default_ruleset()

    result = run_analysis_pipeline(video_path, duration_ms=20_000, fps=30.0, ruleset=ruleset)

    assert len(result.events) > 0
    for event in result.events:
        assert event.model_name.startswith("mock-")
        assert event.model_version.endswith("-mock")

    assert result.score.ruleset_version == ruleset.version
    assert any("unofficial" in w.lower() for w in result.score.warnings)

    for adapter_key, model_id in result.model_versions.items():
        assert "mock-" in model_id or "0.0.0-mock" in model_id, adapter_key


def test_run_analysis_pipeline_kinematics_only_omits_multimodal_model_versions() -> None:
    """Default `feature_fusion_mode="kinematics_only"` must be an exact no-op
    for pre-Prompt-E callers -- no `rgb_encoder`/`string_segmenter`/
    `audio_analyzer` keys should appear in `model_versions`."""
    video_path = Path("/tmp/does-not-need-to-exist-for-mock-adapters.mp4")
    ruleset = default_ruleset()

    result = run_analysis_pipeline(video_path, duration_ms=20_000, fps=30.0, ruleset=ruleset)

    assert "rgb_encoder" not in result.model_versions
    assert "string_segmenter" not in result.model_versions
    assert "audio_analyzer" not in result.model_versions


def test_run_analysis_pipeline_fused_mode_adds_multimodal_model_versions() -> None:
    video_path = Path("/tmp/does-not-need-to-exist-for-mock-adapters.mp4")
    ruleset = default_ruleset()

    result = run_analysis_pipeline(
        video_path,
        duration_ms=20_000,
        fps=30.0,
        ruleset=ruleset,
        feature_fusion_mode="fused",
    )

    for key in ("rgb_encoder", "string_segmenter", "audio_analyzer"):
        assert key in result.model_versions
        assert "mock-" in result.model_versions[key] or "0.0.0-mock" in result.model_versions[key]


def test_run_analysis_pipeline_fused_mode_is_deterministic() -> None:
    video_path = Path("/tmp/does-not-need-to-exist-for-mock-adapters.mp4")
    ruleset = default_ruleset()

    result_a = run_analysis_pipeline(
        video_path, duration_ms=20_000, fps=30.0, ruleset=ruleset, feature_fusion_mode="fused"
    )
    result_b = run_analysis_pipeline(
        video_path, duration_ms=20_000, fps=30.0, ruleset=ruleset, feature_fusion_mode="fused"
    )

    assert result_a.events == result_b.events
    assert result_a.deductions == result_b.deductions
    assert result_a.score == result_b.score
    assert result_a.model_versions == result_b.model_versions


def test_run_analysis_pipeline_fused_and_kinematics_only_both_produce_events() -> None:
    """Fusing in extra modalities must not break temporal event detection --
    the mock temporal event detector still runs against the enriched feature
    timeline and should still emit events."""
    video_path = Path("/tmp/does-not-need-to-exist-for-mock-adapters.mp4")
    ruleset = default_ruleset()

    kinematics_only = run_analysis_pipeline(
        video_path,
        duration_ms=20_000,
        fps=30.0,
        ruleset=ruleset,
        feature_fusion_mode="kinematics_only",
    )
    fused = run_analysis_pipeline(
        video_path, duration_ms=20_000, fps=30.0, ruleset=ruleset, feature_fusion_mode="fused"
    )

    assert len(kinematics_only.events) > 0
    assert len(fused.events) > 0
