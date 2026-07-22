"""Converts this package's internal `EventDetection` into the persisted
`domain.AnalysisEventPrediction` every `TemporalEventDetector` adapter must
return, per `interfaces.TemporalEventDetector.predict`.

Kept separate from `events/types.py` (which stays pure dataclasses, no
`domain` conversion logic) and from `events/decode.py` (which only produces
`EventDetection`s, agnostic of how a caller will persist them) -- mirrors
`perception/evaluation.py`'s `ground_truth_from_dataset_track` being the one
conversion point between an internal representation and a schema owned by
another module.
"""

from __future__ import annotations

from yoyovision_ml.domain import AnalysisEventPrediction, DifficultyBand, EvidenceRef
from yoyovision_ml.events.types import EventDetection


def to_analysis_event_prediction(
    detection: EventDetection, model_name: str
) -> AnalysisEventPrediction:
    """`difficulty_band` is always `UNKNOWN` -- Prompt C's model does not
    predict difficulty, only label/outcome/confidence, so claiming a band
    here would be dishonest. `needs_review` (when the detector's
    `InferenceConfig.uncertainty_action == "flag_review"`) is surfaced as an
    evidence note rather than a dropped field, since `AnalysisEventPrediction`
    has no dedicated review flag -- review routing happens downstream when
    this becomes a persisted `AnalysisEvent` (`domain.ReviewStatus`)."""
    note = (
        "needs_review: low_confidence, flagged for human review" if detection.needs_review else ""
    )
    return AnalysisEventPrediction(
        label=detection.label,
        family=detection.family,
        start_ms=detection.start_ms,
        end_ms=detection.end_ms,
        confidence=detection.confidence,
        outcome=detection.outcome,
        difficulty_band=DifficultyBand.UNKNOWN,
        model_name=model_name,
        model_version=detection.model_version,
        evidence=(EvidenceRef(frame_ms=detection.supporting_frame_range[0], note=note),),
    )
