"""Prompt D "FREESTYLE EVALUATION" section: separate optional estimators for
execution, control, trick diversity, space use and emphasis, music
choreography, music construction, body control -- showmanship stays manual.
Tests `scoring.fe_estimators`."""

from __future__ import annotations

from conftest import make_event_prediction, make_feature_set

from yoyovision_ml.domain import EventFamily, Outcome
from yoyovision_ml.perception.features import (
    FEATURE_LEFT_ELBOW_ANGLE_DEG,
    FEATURE_RIGHT_ELBOW_ANGLE_DEG,
    FEATURE_STAGE_X,
    FEATURE_STAGE_Y,
)
from yoyovision_ml.scoring.fe_estimators import (
    UNSUPPORTED_CATEGORIES,
    estimate_all,
    estimate_body_control,
    estimate_control,
    estimate_execution,
    estimate_music_choreography,
    estimate_music_construction,
    estimate_space_use_emphasis,
    estimate_trick_diversity,
)

_ALL_ESTIMATORS_SHARE_THESE_INVARIANTS = (
    estimate_execution([]),
    estimate_control([]),
    estimate_trick_diversity([]),
    estimate_space_use_emphasis(None),
    estimate_body_control(None),
    estimate_music_choreography(),
    estimate_music_construction(),
)


def test_every_estimate_exposes_confidence_features_version_and_warning() -> None:
    """Prompt D: "All automatically generated FE categories must expose:
    confidence, supporting features, model version, warning that artistic
    scoring is subjective." Checked once, generically, across every
    estimator's empty-input output. The two music categories substitute a
    more specific "no audio-analysis stage" warning instead of the generic
    subjectivity notice (see `fe_estimators._NO_AUDIO_WARNING`) since they
    never estimate a value at all -- both count as satisfying the
    requirement, so they are exempted from the "subjective" substring check
    but still checked for every other field."""
    for estimate in _ALL_ESTIMATORS_SHARE_THESE_INVARIANTS:
        assert isinstance(estimate.confidence, float)
        assert isinstance(estimate.supporting_features, dict)
        assert estimate.model_version
        assert estimate.warning
        if estimate.category not in UNSUPPORTED_CATEGORIES:
            assert "subjective" in estimate.warning.lower()
        else:
            assert "manually" in estimate.warning.lower()


def test_estimate_all_never_returns_showmanship_key() -> None:
    """Prompt D: "Keep showmanship manual by default" -- there is no
    `estimate_showmanship`, so `estimate_all` must never synthesize one."""
    estimates = estimate_all([], None)
    assert "showmanship" not in estimates
    assert set(estimates.keys()) == {
        "execution",
        "control",
        "trick_diversity",
        "space_use_emphasis",
        "music_choreography",
        "music_construction",
        "body_control",
    }


def test_execution_estimate_is_none_with_no_attempts() -> None:
    estimate = estimate_execution([])
    assert estimate.value is None
    assert estimate.confidence == 0.0


def test_execution_estimate_reflects_success_rate() -> None:
    events = [
        make_event_prediction(label="mount_1", outcome=Outcome.SUCCESS),
        make_event_prediction(label="mount_2", outcome=Outcome.SUCCESS),
        make_event_prediction(label="mount_3", outcome=Outcome.MISS),
        make_event_prediction(label="mount_4", outcome=Outcome.MISS),
    ]
    estimate = estimate_execution(events)
    assert estimate.value == 5.0  # 50% success rate scaled to 0-10
    assert estimate.supporting_features["attempt_count"] == 4.0


def test_execution_estimate_ignores_non_positive_families() -> None:
    events = [
        make_event_prediction(
            label="miss_1", family=EventFamily.CONTROL_MISS, outcome=Outcome.MISS
        )
    ]
    estimate = estimate_execution(events)
    assert estimate.value is None


def test_control_estimate_penalizes_control_misses() -> None:
    events = [
        make_event_prediction(label="mount_1", outcome=Outcome.SUCCESS),
        make_event_prediction(
            label="control_miss_1", family=EventFamily.CONTROL_MISS, outcome=Outcome.MISS
        ),
    ]
    estimate = estimate_control(events)
    assert estimate.value == 5.0  # 1 miss out of 2 attempts -> 50% -> 5.0
    assert estimate.supporting_features["control_miss_count"] == 1.0


def test_control_estimate_full_score_with_no_misses() -> None:
    events = [make_event_prediction(label="mount_1", outcome=Outcome.SUCCESS)]
    estimate = estimate_control(events)
    assert estimate.value == 10.0


def test_trick_diversity_counts_distinct_successful_families() -> None:
    events = [
        make_event_prediction(label="mount_1", family=EventFamily.MOUNT, outcome=Outcome.SUCCESS),
        make_event_prediction(label="hop_1", family=EventFamily.HOP, outcome=Outcome.SUCCESS),
        # A miss doesn't count towards diversity even though the family is positive.
        make_event_prediction(
            label="suicide_1", family=EventFamily.SUICIDE, outcome=Outcome.MISS
        ),
    ]
    estimate = estimate_trick_diversity(events)
    assert estimate.supporting_features["distinct_families_landed"] == 2.0


def test_trick_diversity_zero_with_no_events() -> None:
    estimate = estimate_trick_diversity([])
    assert estimate.value == 0.0
    assert estimate.confidence == 0.0


def test_space_use_emphasis_none_without_features() -> None:
    assert estimate_space_use_emphasis(None).value is None
    empty_features = make_feature_set(frame_count=0, values_by_frame=[])
    assert estimate_space_use_emphasis(empty_features).value is None


def test_space_use_emphasis_none_when_stage_position_missing() -> None:
    features = make_feature_set(frame_count=5, values_by_frame=[{} for _ in range(5)])
    estimate = estimate_space_use_emphasis(features)
    assert estimate.value is None


def test_space_use_emphasis_uses_stage_position_spread() -> None:
    values = [
        {FEATURE_STAGE_X: x, FEATURE_STAGE_Y: y}
        for x, y in [(-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)]
    ]
    features = make_feature_set(frame_count=4, values_by_frame=values)
    estimate = estimate_space_use_emphasis(features)
    assert estimate.value is not None
    assert estimate.value > 0.0
    assert estimate.supporting_features["frame_count"] == 4.0


def test_body_control_none_without_features() -> None:
    assert estimate_body_control(None).value is None


def test_body_control_none_with_too_few_samples() -> None:
    features = make_feature_set(
        frame_count=1, values_by_frame=[{FEATURE_LEFT_ELBOW_ANGLE_DEG: 90.0}]
    )
    estimate = estimate_body_control(features)
    assert estimate.value is None


def test_body_control_high_score_for_zero_jitter() -> None:
    values = [
        {FEATURE_LEFT_ELBOW_ANGLE_DEG: 90.0, FEATURE_RIGHT_ELBOW_ANGLE_DEG: 90.0}
        for _ in range(10)
    ]
    features = make_feature_set(frame_count=10, values_by_frame=values)
    estimate = estimate_body_control(features)
    assert estimate.value == 10.0


def test_body_control_lower_score_for_high_jitter() -> None:
    values = [
        {FEATURE_LEFT_ELBOW_ANGLE_DEG: angle, FEATURE_RIGHT_ELBOW_ANGLE_DEG: angle}
        for angle in [0.0, 180.0, 0.0, 180.0, 0.0, 180.0]
    ]
    features = make_feature_set(frame_count=6, values_by_frame=values)
    estimate = estimate_body_control(features)
    assert estimate.value is not None
    assert estimate.value < 10.0


def test_music_categories_always_return_none_never_a_guess() -> None:
    """Prompt D FE section, via this module's own docstring: "This pipeline
    has no audio-analysis stage ... always None, never a guess." Music
    estimates take no arguments at all -- there is nothing to vary."""
    choreography = estimate_music_choreography()
    construction = estimate_music_construction()
    assert choreography.value is None
    assert construction.value is None
    assert choreography.confidence == 0.0
    assert construction.confidence == 0.0


def test_unsupported_categories_matches_music_categories() -> None:
    assert UNSUPPORTED_CATEGORIES == ("music_choreography", "music_construction")
