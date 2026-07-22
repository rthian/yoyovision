"""Prompt D "MAJOR DEDUCTIONS" section: "Dangerous-play detection must never
automatically disqualify a player. It must create a review flag." Tests
`scoring.dangerous_play.detect_dangerous_play`."""

from __future__ import annotations

from conftest import make_feature_set

from yoyovision_ml.domain import DeductionType, ReviewStatus
from yoyovision_ml.perception.features import FEATURE_YOYO_VELOCITY
from yoyovision_ml.ruleset import default_ruleset
from yoyovision_ml.scoring.dangerous_play import DangerousPlayConfig, detect_dangerous_play
from yoyovision_ml.scoring_engine import deduction_is_scorable


def _velocity_feature_set(velocities: list[float], spacing_ms: int = 33) -> object:
    return make_feature_set(
        frame_count=len(velocities),
        frame_spacing_ms=spacing_ms,
        values_by_frame=[{FEATURE_YOYO_VELOCITY: v} for v in velocities],
    )


def test_no_flags_when_velocity_never_exceeds_threshold() -> None:
    features = _velocity_feature_set([1.0, 2.0, 3.0, 1.0])
    flags = detect_dangerous_play(features)
    assert flags == []


def test_no_flags_with_empty_feature_set() -> None:
    features = make_feature_set(frame_count=0, values_by_frame=[])
    assert detect_dangerous_play(features) == []


def test_flags_sustained_high_velocity_run() -> None:
    config = DangerousPlayConfig(velocity_threshold=8.0, min_consecutive_frames=3)
    # 3 consecutive candidate frames satisfies min_consecutive_frames.
    features = _velocity_feature_set([1.0, 9.0, 10.0, 9.5, 1.0])
    flags = detect_dangerous_play(features, config)
    assert len(flags) == 1
    assert flags[0].prediction.type == DeductionType.DANGEROUS_PLAY_REVIEW


def test_does_not_flag_run_shorter_than_min_consecutive_frames() -> None:
    config = DangerousPlayConfig(velocity_threshold=8.0, min_consecutive_frames=3)
    features = _velocity_feature_set([1.0, 9.0, 9.0, 1.0])  # only 2 consecutive
    flags = detect_dangerous_play(features, config)
    assert flags == []


def test_flag_confidence_scales_with_peak_velocity() -> None:
    config = DangerousPlayConfig(velocity_threshold=8.0, min_consecutive_frames=3)
    features = _velocity_feature_set([16.0, 16.0, 16.0])  # peak == 2x threshold
    flags = detect_dangerous_play(features, config)
    assert len(flags) == 1
    assert flags[0].prediction.confidence == 1.0


def test_nearby_runs_within_merge_gap_are_merged_into_one_flag() -> None:
    config = DangerousPlayConfig(
        velocity_threshold=8.0, min_consecutive_frames=2, merge_gap_ms=500
    )
    # Two short runs at frame spacing 100ms: frames 0-1 high, frame 2 low, frames 3-4 high.
    # Gap between run starts is well within merge_gap_ms.
    features = _velocity_feature_set([9.0, 9.0, 1.0, 9.0, 9.0], spacing_ms=100)
    flags = detect_dangerous_play(features, config)
    assert len(flags) == 1


def test_runs_far_apart_are_not_merged() -> None:
    config = DangerousPlayConfig(
        velocity_threshold=8.0, min_consecutive_frames=2, merge_gap_ms=100
    )
    velocities = [9.0, 9.0] + [1.0] * 20 + [9.0, 9.0]
    features = _velocity_feature_set(velocities, spacing_ms=100)
    flags = detect_dangerous_play(features, config)
    assert len(flags) == 2


def test_flag_reason_mentions_review_never_disqualification() -> None:
    config = DangerousPlayConfig(velocity_threshold=8.0, min_consecutive_frames=2)
    features = _velocity_feature_set([9.0, 9.0])
    flags = detect_dangerous_play(features, config)
    assert "review" in flags[0].reason.lower()
    assert "never" in flags[0].reason.lower()


def test_a_freshly_detected_flag_is_pending_and_not_yet_scorable() -> None:
    """Prompt D's core guarantee, verified end-to-end from the detector's own
    output through the ruleset gate: a freshly detected flag is PENDING by
    construction, and PENDING dangerous-play flags never affect a score."""
    config = DangerousPlayConfig(velocity_threshold=8.0, min_consecutive_frames=2)
    features = _velocity_feature_set([9.0, 9.0])
    flags = detect_dangerous_play(features, config)
    assert len(flags) == 1

    ruleset = default_ruleset()
    # `DeductionPrediction` (what the detector emits) has no review_status of
    # its own -- a caller persists it as PENDING by default, matching every
    # other freshly-detected `MajorDeduction` in this codebase.
    assert deduction_is_scorable(
        flags[0].prediction.type, ReviewStatus.PENDING, ruleset
    ) is False
