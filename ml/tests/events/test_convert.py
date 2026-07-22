from __future__ import annotations

from yoyovision_ml.domain import DifficultyBand, EventFamily, Outcome
from yoyovision_ml.events.convert import to_analysis_event_prediction
from yoyovision_ml.events.types import EventDetection


def _detection(needs_review: bool = False) -> EventDetection:
    return EventDetection(
        label="hop",
        family=EventFamily.HOP,
        start_ms=100,
        end_ms=300,
        outcome=Outcome.SUCCESS,
        confidence=0.87,
        model_version="trick-event-tcn-1.0",
        supporting_frame_range=(100, 300),
        needs_review=needs_review,
    )


def test_to_analysis_event_prediction_maps_every_core_field() -> None:
    prediction = to_analysis_event_prediction(_detection(), model_name="trick-event-tcn")
    assert prediction.label == "hop"
    assert prediction.family == EventFamily.HOP
    assert prediction.start_ms == 100
    assert prediction.end_ms == 300
    assert prediction.confidence == 0.87
    assert prediction.outcome == Outcome.SUCCESS
    assert prediction.model_name == "trick-event-tcn"
    assert prediction.model_version == "trick-event-tcn-1.0"


def test_to_analysis_event_prediction_never_claims_a_difficulty_band() -> None:
    prediction = to_analysis_event_prediction(_detection(), model_name="trick-event-tcn")
    assert prediction.difficulty_band == DifficultyBand.UNKNOWN


def test_to_analysis_event_prediction_evidence_points_at_supporting_frame_range_start() -> None:
    prediction = to_analysis_event_prediction(_detection(), model_name="trick-event-tcn")
    assert len(prediction.evidence) == 1
    assert prediction.evidence[0].frame_ms == 100


def test_to_analysis_event_prediction_notes_needs_review_when_flagged() -> None:
    prediction = to_analysis_event_prediction(
        _detection(needs_review=True), model_name="trick-event-tcn"
    )
    assert "needs_review" in prediction.evidence[0].note


def test_to_analysis_event_prediction_leaves_note_empty_when_not_flagged() -> None:
    prediction = to_analysis_event_prediction(
        _detection(needs_review=False), model_name="trick-event-tcn"
    )
    assert prediction.evidence[0].note == ""
