from __future__ import annotations

from datetime import UTC, datetime

from yoyovision_ml.adapters_registry import create_temporal_event_detector
from yoyovision_ml.dataset.schema import AnnotationProvenance, TrickEventAnnotation
from yoyovision_ml.domain import EventFamily, FeatureFrame, FeatureSet, Outcome, Source
from yoyovision_ml.events.baselines import MajorityClassEventDetector, ThresholdRuleEventDetector
from yoyovision_ml.events.types import TrainingSample
from yoyovision_ml.perception.features import (
    FEATURE_HAND_DISTANCE,
    FEATURE_YOYO_DIRECTION_DEG,
    FEATURE_YOYO_VELOCITY,
)


def _event(
    family: EventFamily,
    outcome: Outcome,
    start_ms: int = 0,
    end_ms: int = 100,
    event_id: str = "e0",
) -> TrickEventAnnotation:
    return TrickEventAnnotation(
        event_id=event_id,
        label=family.value,
        family=family,
        start_ms=start_ms,
        end_ms=end_ms,
        outcome=outcome,
        provenance=AnnotationProvenance(
            annotator_id="test", source=Source.HUMAN, annotated_at=datetime(2024, 1, 1, tzinfo=UTC)
        ),
    )


def _sample(events: list[TrickEventAnnotation], player_id: str = "p1") -> TrainingSample:
    features = FeatureSet(frames=(), feature_names=(), fps=30.0)
    return TrainingSample(
        video_id="v1", player_id=player_id, features=features, trick_events=tuple(events)
    )


def _feature_set_with_rows(rows: list[tuple[int, dict[str, float]]]) -> FeatureSet:
    frames = tuple(FeatureFrame(frame_ms=ms, values=values) for ms, values in rows)
    return FeatureSet(frames=frames, feature_names=(), fps=10.0)


# --------------------------------------------------------------------------- #
# MajorityClassEventDetector
# --------------------------------------------------------------------------- #
def test_majority_class_detector_default_is_unknown_and_uncertain() -> None:
    detector = MajorityClassEventDetector()
    assert detector._family == EventFamily.UNKNOWN_TECHNICAL_ELEMENT
    assert detector._outcome == Outcome.UNCERTAIN


def test_majority_class_detector_fit_with_no_samples_falls_back_to_default() -> None:
    detector = MajorityClassEventDetector.fit([])
    assert detector._family == EventFamily.UNKNOWN_TECHNICAL_ELEMENT
    assert detector._outcome == Outcome.UNCERTAIN


def test_majority_class_detector_fit_picks_most_frequent_family_and_outcome() -> None:
    samples = [
        _sample(
            [
                _event(EventFamily.HOP, Outcome.SUCCESS, event_id="e0"),
                _event(EventFamily.HOP, Outcome.SUCCESS, event_id="e1"),
                _event(EventFamily.SLACK, Outcome.MISS, event_id="e2"),
            ]
        )
    ]
    detector = MajorityClassEventDetector.fit(samples)
    assert detector._family == EventFamily.HOP
    assert detector._outcome == Outcome.SUCCESS
    assert detector._confidence == 2 / 3


def test_majority_class_detector_fit_skips_equipment_families() -> None:
    """`yoyo_stop` is outside Prompt C's 20 classes -- must never be chosen
    as the majority family."""
    samples = [
        _sample(
            [
                _event(EventFamily.YOYO_STOP, Outcome.MISS, event_id="e0"),
                _event(EventFamily.YOYO_STOP, Outcome.MISS, event_id="e1"),
                _event(EventFamily.HOP, Outcome.SUCCESS, event_id="e2"),
            ]
        )
    ]
    detector = MajorityClassEventDetector.fit(samples)
    assert detector._family == EventFamily.HOP


def test_majority_class_detector_predict_returns_empty_for_no_frames() -> None:
    detector = MajorityClassEventDetector(EventFamily.HOP, Outcome.SUCCESS, 0.5)
    predictions, deductions = detector.predict(FeatureSet(frames=(), feature_names=(), fps=30.0))
    assert predictions == []
    assert deductions == []


def test_majority_class_detector_predict_spans_the_entire_clip() -> None:
    detector = MajorityClassEventDetector(EventFamily.HOP, Outcome.SUCCESS, 0.75)
    features = _feature_set_with_rows([(0, {}), (100, {}), (200, {})])
    predictions, deductions = detector.predict(features)
    assert deductions == []
    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.family == EventFamily.HOP
    assert prediction.outcome == Outcome.SUCCESS
    assert prediction.start_ms == 0
    assert prediction.end_ms == 200
    assert prediction.confidence == 0.75
    assert prediction.model_name == MajorityClassEventDetector.model_name


def test_majority_class_detector_predict_returns_empty_for_single_frame_clip() -> None:
    """A single-frame clip has `end_ms <= start_ms`, so there is no non-zero
    span to predict an event over."""
    detector = MajorityClassEventDetector(EventFamily.HOP, Outcome.SUCCESS, 0.5)
    features = _feature_set_with_rows([(0, {})])
    predictions, _deductions = detector.predict(features)
    assert predictions == []


def test_majority_detector_is_registered_under_majority() -> None:
    detector = create_temporal_event_detector("majority")
    assert isinstance(detector, MajorityClassEventDetector)


# --------------------------------------------------------------------------- #
# ThresholdRuleEventDetector
# --------------------------------------------------------------------------- #
def _threshold_rule_features() -> FeatureSet:
    """11 frames, 100ms apart. Frames 1-2 simulate a fast vertical yo-yo
    motion (hop rule); frames 4-5 simulate a very fast horizontal motion
    (whip-catch rule, since it is *not* vertical so the hop rule does not
    also fire); frames 7-8 simulate near-zero velocity with the hand far
    from the yo-yo (slack rule). All other frames are neutral (fire no
    rule)."""
    neutral = {
        FEATURE_YOYO_VELOCITY: 0.2,
        FEATURE_YOYO_DIRECTION_DEG: 45.0,
        FEATURE_HAND_DISTANCE: 0.1,
    }
    rows: list[tuple[int, dict[str, float]]] = []
    for i in range(11):
        frame_ms = i * 100
        if i in (1, 2):
            values = {
                FEATURE_YOYO_VELOCITY: 0.6,
                FEATURE_YOYO_DIRECTION_DEG: 90.0,
                FEATURE_HAND_DISTANCE: 0.1,
            }
        elif i in (4, 5):
            values = {
                FEATURE_YOYO_VELOCITY: 0.9,
                FEATURE_YOYO_DIRECTION_DEG: 0.0,
                FEATURE_HAND_DISTANCE: 0.1,
            }
        elif i in (7, 8):
            values = {
                FEATURE_YOYO_VELOCITY: 0.01,
                FEATURE_YOYO_DIRECTION_DEG: 45.0,
                FEATURE_HAND_DISTANCE: 0.5,
            }
        else:
            values = dict(neutral)
        rows.append((frame_ms, values))
    return _feature_set_with_rows(rows)


def test_threshold_rule_detector_predict_returns_empty_for_no_frames() -> None:
    detector = ThresholdRuleEventDetector()
    predictions, deductions = detector.predict(FeatureSet(frames=(), feature_names=(), fps=30.0))
    assert predictions == []
    assert deductions == []


def test_threshold_rule_detector_fires_hop_on_fast_vertical_motion() -> None:
    detector = ThresholdRuleEventDetector()
    predictions, _deductions = detector.predict(_threshold_rule_features())
    hops = [p for p in predictions if p.family == EventFamily.HOP]
    assert len(hops) == 1
    assert hops[0].start_ms == 100
    assert hops[0].end_ms == 300


def test_threshold_rule_detector_fires_whip_catch_on_very_fast_horizontal_motion() -> None:
    detector = ThresholdRuleEventDetector()
    predictions, _deductions = detector.predict(_threshold_rule_features())
    whips = [p for p in predictions if p.family == EventFamily.WHIP_CATCH]
    assert len(whips) == 1
    assert whips[0].start_ms == 400
    assert whips[0].end_ms == 600


def test_threshold_rule_detector_fires_slack_on_near_zero_velocity_and_far_hand() -> None:
    detector = ThresholdRuleEventDetector()
    predictions, _deductions = detector.predict(_threshold_rule_features())
    slacks = [p for p in predictions if p.family == EventFamily.SLACK]
    assert len(slacks) == 1
    assert slacks[0].start_ms == 700
    assert slacks[0].end_ms == 900


def test_threshold_rule_detector_predictions_are_always_uncertain_outcome() -> None:
    detector = ThresholdRuleEventDetector()
    predictions, _deductions = detector.predict(_threshold_rule_features())
    assert predictions  # sanity: rules actually fired
    assert all(p.outcome == Outcome.UNCERTAIN for p in predictions)


def test_threshold_rule_detector_never_returns_deductions() -> None:
    detector = ThresholdRuleEventDetector()
    _predictions, deductions = detector.predict(_threshold_rule_features())
    assert deductions == []


def test_rules_detector_is_registered_under_rules() -> None:
    detector = create_temporal_event_detector("rules")
    assert isinstance(detector, ThresholdRuleEventDetector)
