"""Tests for `yoyovision_ml.inference.monitoring`."""

from __future__ import annotations

import pytest
from yoyovision_ml.domain import (
    AnalysisEventPrediction,
    BoundingBox,
    DifficultyBand,
    EventFamily,
    Outcome,
    Track,
    VisibilityState,
)
from yoyovision_ml.inference.monitoring import ReferenceBaseline, compute_monitoring_signals


def _event(family: EventFamily, confidence: float) -> AnalysisEventPrediction:
    return AnalysisEventPrediction(
        label=family.value,
        family=family,
        start_ms=0,
        end_ms=100,
        confidence=confidence,
        outcome=Outcome.SUCCESS,
        difficulty_band=DifficultyBand.BASIC,
        model_name="test-model",
        model_version="0.0.0-test",
    )


def _track(*, interpolated: bool, confidence: float) -> Track:
    return Track(
        track_id="t1",
        frame_ms=0,
        bbox=BoundingBox(x=0.0, y=0.0, width=0.1, height=0.1),
        confidence=confidence,
        class_label="yoyo",
        visibility=VisibilityState.VISIBLE,
        interpolated=interpolated,
    )


def test_compute_monitoring_signals_with_no_events_or_tracks() -> None:
    signals = compute_monitoring_signals([], [])

    assert signals.class_counts == {}
    assert signals.avg_confidence == 0.0
    assert signals.failed_track_rate == 0.0


def test_compute_monitoring_signals_counts_classes_and_confidence() -> None:
    events = [_event(EventFamily.MOUNT, 0.9), _event(EventFamily.HOP, 0.3)]

    signals = compute_monitoring_signals(events, [])

    assert signals.class_counts == {"mount": 1, "hop": 1}
    assert signals.avg_confidence == 0.6
    assert signals.low_confidence_rate == 0.5


def test_compute_monitoring_signals_failed_track_rate() -> None:
    tracks = [
        _track(interpolated=False, confidence=0.9),
        _track(interpolated=True, confidence=0.9),
        _track(interpolated=False, confidence=0.1),
    ]

    signals = compute_monitoring_signals([], tracks)

    assert signals.failed_track_rate == pytest.approx(2 / 3)


def test_compute_monitoring_signals_drift_scores_against_reference() -> None:
    events = [_event(EventFamily.MOUNT, 0.8), _event(EventFamily.MOUNT, 0.8)]
    reference = ReferenceBaseline(class_frequencies={"mount": 0.5, "hop": 0.5}, avg_confidence=0.8)

    signals = compute_monitoring_signals(events, [], reference=reference)

    assert signals.class_drift_score == pytest.approx(0.5)
    assert signals.confidence_drift_score == pytest.approx(0.0)
