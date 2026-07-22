"""Tests for `yoyovision_ml.inference.report`."""

from __future__ import annotations

from pathlib import Path

from yoyovision_ml.inference.device import resolve_device
from yoyovision_ml.inference.report import generate_human_readable_report
from yoyovision_ml.inference.timing import StageTimings
from yoyovision_ml.pipeline import PipelineResult, run_analysis_pipeline
from yoyovision_ml.ruleset import default_ruleset


def _fake_timings(result: PipelineResult) -> StageTimings:
    timings = StageTimings()
    timings.durations_ms = dict(result.stage_durations_ms)
    return timings


def test_generate_human_readable_report_contains_no_local_paths() -> None:
    video_path = Path("/tmp/some/absolute/local/path/video.mp4")
    result = run_analysis_pipeline(
        video_path, duration_ms=20_000, fps=30.0, ruleset=default_ruleset()
    )
    device_info = resolve_device("cpu")

    report = generate_human_readable_report(
        job_id="job-123",
        video_filename=video_path.name,
        pipeline_version="0.1.0-test",
        result=result,
        timings=_fake_timings(result),
        device_info=device_info,
        runtime_versions=result.runtime_versions,
    )

    assert str(video_path) not in report
    assert str(video_path.parent) not in report
    assert video_path.name in report


def test_generate_human_readable_report_includes_score_and_model_versions() -> None:
    video_path = Path("clip.mp4")
    result = run_analysis_pipeline(
        video_path, duration_ms=20_000, fps=30.0, ruleset=default_ruleset()
    )
    device_info = resolve_device("cpu")

    report = generate_human_readable_report(
        job_id="job-456",
        video_filename=video_path.name,
        pipeline_version="0.1.0-test",
        result=result,
        timings=_fake_timings(result),
        device_info=device_info,
        runtime_versions=result.runtime_versions,
        monitoring=result.monitoring,
    )

    assert "job-456" in report
    assert f"{result.score.final_score:.2f}" in report
    for model_id in result.model_versions.values():
        assert model_id in report
