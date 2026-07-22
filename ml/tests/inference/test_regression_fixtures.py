"""Prompt F regression fixtures: runs the full (mock-adapter) pipeline
against the committed, consented synthetic sample dataset
(`ml/sample_data/dataset_v1/`, see that directory's `README.md` -- every
video file under it is placeholder text, not real footage) and pins the
resulting score/event counts.

This is a drift guard, not a model-quality test: if a future refactor of
`pipeline.py`, the scoring engine, or the mock adapters accidentally changes
output for the same inputs, this test fails loudly instead of the change
going unnoticed. Update `_EXPECTED` deliberately (with a changelog note)
whenever a real, intended pipeline behavior change ships.
"""

from __future__ import annotations

from pathlib import Path

from yoyovision_ml.dataset.io import load_dataset
from yoyovision_ml.pipeline import run_analysis_pipeline
from yoyovision_ml.ruleset import default_ruleset

SAMPLE_DATASET_DIR = Path(__file__).parent.parent.parent / "sample_data" / "dataset_v1"

#: One reference snapshot per sample video id, computed once against the
#: current deterministic mock adapters + default ruleset.
_EXPECTED: dict[str, dict[str, float | int]] = {
    "sample_video_001": {"event_count": 13, "deduction_count": 1, "final_score": 2.2},
    "sample_video_002": {"event_count": 13, "deduction_count": 1, "final_score": 2.2},
    "sample_video_003": {"event_count": 13, "deduction_count": 1, "final_score": 2.2},
}


def _sample_videos() -> dict[str, object]:
    _, records = load_dataset(SAMPLE_DATASET_DIR)
    by_video_id: dict[str, object] = {}
    for record in records:
        by_video_id.setdefault(record.video.video_id, record.video)
    return by_video_id


def test_regression_fixture_dataset_covers_every_expected_video() -> None:
    videos = _sample_videos()
    assert set(videos) == set(_EXPECTED)


def test_mock_pipeline_output_matches_pinned_regression_snapshot() -> None:
    ruleset = default_ruleset()
    videos = _sample_videos()

    for video_id, expected in _EXPECTED.items():
        video = videos[video_id]
        video_path = SAMPLE_DATASET_DIR / video.relative_path  # type: ignore[attr-defined]
        result = run_analysis_pipeline(
            video_path,
            duration_ms=video.duration_ms,  # type: ignore[attr-defined]
            fps=video.source_fps,  # type: ignore[attr-defined]
            ruleset=ruleset,
        )

        assert len(result.events) == expected["event_count"], video_id
        assert len(result.deductions) == expected["deduction_count"], video_id
        assert result.score.final_score == expected["final_score"], video_id


def test_mock_pipeline_output_is_deterministic_across_repeated_runs() -> None:
    """Same regression guard from a different angle: two runs against the
    same fixture video must be bit-for-bit identical, independent of
    whatever the pinned snapshot above says."""
    ruleset = default_ruleset()
    video = _sample_videos()["sample_video_001"]
    video_path = SAMPLE_DATASET_DIR / video.relative_path  # type: ignore[attr-defined]

    result_a = run_analysis_pipeline(
        video_path, duration_ms=video.duration_ms, fps=video.source_fps, ruleset=ruleset  # type: ignore[attr-defined]
    )
    result_b = run_analysis_pipeline(
        video_path, duration_ms=video.duration_ms, fps=video.source_fps, ruleset=ruleset  # type: ignore[attr-defined]
    )

    assert result_a.events == result_b.events
    assert result_a.deductions == result_b.deductions
    assert result_a.score == result_b.score
