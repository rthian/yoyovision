"""Prompt D "FREESTYLE EVALUATION" section: "Implement separate optional
estimators for: execution, control, trick diversity, space use and
emphasis, music choreography, music construction, body control. Keep
showmanship manual by default."

Every estimator is a modest, hand-crafted heuristic over already-detected
events and/or Prompt B kinematic features -- explicitly NOT a trained model
(no Freestyle-Evaluation-category training labels exist in this
repository). Per the prompt's explicit requirement, every estimate carries
its own confidence, the features that produced it, a `model_version`, and a
warning that artistic scoring is inherently subjective.

`showmanship` has no estimator function in this module at all -- it is
manual-only by design, and `estimate_all` never returns a "showmanship" key.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from yoyovision_ml.domain import (
    POSITIVE_EVENT_FAMILIES,
    AnalysisEventPrediction,
    EventFamily,
    FeatureSet,
    Outcome,
)
from yoyovision_ml.perception.features import (
    FEATURE_LEFT_ELBOW_ANGLE_DEG,
    FEATURE_RIGHT_ELBOW_ANGLE_DEG,
    FEATURE_STAGE_X,
    FEATURE_STAGE_Y,
)
from yoyovision_ml.scoring.types import FreestyleEvaluationEstimate

MODEL_NAME = "heuristic-fe-estimator"
MODEL_VERSION = "0.1.0-heuristic"

_SUBJECTIVITY_WARNING = (
    "Automatically generated -- artistic/subjective scoring is inherently "
    "subjective and this estimate is not a substitute for human judgment."
)
_NO_AUDIO_WARNING = (
    "This pipeline has no audio-analysis stage; music categories cannot be "
    "estimated from video/pose features alone and must be entered manually "
    "by a human judge. This estimate is always None, never a guess."
)


def _empty_estimate(category: str, warning: str) -> FreestyleEvaluationEstimate:
    return FreestyleEvaluationEstimate(
        category=category,
        value=None,
        confidence=0.0,
        supporting_features={},
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        warning=warning,
    )


def _confidence_from_sample_size(n: int, saturate_at: int) -> float:
    """More samples -> more confidence, capped at 1.0; 0 samples -> 0.0."""
    if n <= 0:
        return 0.0
    return round(min(1.0, n / saturate_at), 3)


def estimate_execution(events: Sequence[AnalysisEventPrediction]) -> FreestyleEvaluationEstimate:
    """Success rate among attempted positive technical elements, scaled to
    the 0-10 Freestyle Evaluation range."""
    attempts = [e for e in events if e.family in POSITIVE_EVENT_FAMILIES]
    if not attempts:
        return _empty_estimate(
            "execution", "No technical elements detected. " + _SUBJECTIVITY_WARNING
        )
    successes = sum(1 for e in attempts if e.outcome == Outcome.SUCCESS)
    success_rate = successes / len(attempts)
    return FreestyleEvaluationEstimate(
        category="execution",
        value=round(success_rate * 10.0, 3),
        confidence=_confidence_from_sample_size(len(attempts), saturate_at=8),
        supporting_features={"attempt_count": float(len(attempts)), "success_rate": success_rate},
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        warning=_SUBJECTIVITY_WARNING,
    )


def estimate_control(events: Sequence[AnalysisEventPrediction]) -> FreestyleEvaluationEstimate:
    """Inverse rate of `control_miss` among attempted elements -- the family
    this category most directly maps to."""
    attempts = [
        e
        for e in events
        if e.family in POSITIVE_EVENT_FAMILIES or e.family == EventFamily.CONTROL_MISS
    ]
    if not attempts:
        return _empty_estimate(
            "control", "No technical elements detected. " + _SUBJECTIVITY_WARNING
        )
    control_misses = sum(1 for e in events if e.family == EventFamily.CONTROL_MISS)
    control_rate = max(0.0, 1.0 - (control_misses / len(attempts)))
    return FreestyleEvaluationEstimate(
        category="control",
        value=round(control_rate * 10.0, 3),
        confidence=_confidence_from_sample_size(len(attempts), saturate_at=8),
        supporting_features={
            "attempt_count": float(len(attempts)),
            "control_miss_count": float(control_misses),
        },
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        warning=_SUBJECTIVITY_WARNING,
    )


def estimate_trick_diversity(
    events: Sequence[AnalysisEventPrediction],
) -> FreestyleEvaluationEstimate:
    """Share of the 16 positive trick families landed successfully at least
    once during the routine."""
    successful_families = {
        e.family
        for e in events
        if e.family in POSITIVE_EVENT_FAMILIES and e.outcome == Outcome.SUCCESS
    }
    total_positive_families = len(POSITIVE_EVENT_FAMILIES)
    diversity_ratio = len(successful_families) / total_positive_families
    return FreestyleEvaluationEstimate(
        category="trick_diversity",
        value=round(diversity_ratio * 10.0, 3),
        confidence=_confidence_from_sample_size(len(events), saturate_at=12),
        supporting_features={
            "distinct_families_landed": float(len(successful_families)),
            "total_positive_families": float(total_positive_families),
        },
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        warning=_SUBJECTIVITY_WARNING,
    )


def _feature_column(features: FeatureSet, name: str) -> np.ndarray:
    return np.array([frame.values.get(name, np.nan) for frame in features.frames], dtype=float)


def estimate_space_use_emphasis(features: FeatureSet | None) -> FreestyleEvaluationEstimate:
    """Uses the yo-yo's body-relative `stage_x`/`stage_y` spread (Prompt B's
    "body-centered stage position" features) as a rough proxy for how much
    of the available space the player uses."""
    if features is None or not features.frames:
        return _empty_estimate(
            "space_use_emphasis", "No kinematic features available. " + _SUBJECTIVITY_WARNING
        )
    stage_x = _feature_column(features, FEATURE_STAGE_X)
    stage_y = _feature_column(features, FEATURE_STAGE_Y)
    valid = ~(np.isnan(stage_x) | np.isnan(stage_y))
    if not valid.any():
        return _empty_estimate(
            "space_use_emphasis",
            "No stage-position features available. " + _SUBJECTIVITY_WARNING,
        )
    spread = float(np.std(stage_x[valid]) + np.std(stage_y[valid]))
    # Heuristic squashing so an arbitrary-unit spread maps onto 0-10.
    value = float(min(10.0, spread * 5.0))
    return FreestyleEvaluationEstimate(
        category="space_use_emphasis",
        value=round(value, 3),
        confidence=_confidence_from_sample_size(int(valid.sum()), saturate_at=200),
        supporting_features={
            "stage_position_spread": spread,
            "frame_count": float(int(valid.sum())),
        },
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        warning=_SUBJECTIVITY_WARNING,
    )


def estimate_body_control(features: FeatureSet | None) -> FreestyleEvaluationEstimate:
    """Uses low frame-to-frame elbow-angle jitter as a rough proxy for
    smooth, controlled body movement -- lower jitter maps to a higher score.
    """
    if features is None or not features.frames:
        return _empty_estimate(
            "body_control", "No kinematic features available. " + _SUBJECTIVITY_WARNING
        )
    left = _feature_column(features, FEATURE_LEFT_ELBOW_ANGLE_DEG)
    right = _feature_column(features, FEATURE_RIGHT_ELBOW_ANGLE_DEG)
    combined = np.concatenate([left[~np.isnan(left)], right[~np.isnan(right)]])
    if combined.size < 2:
        return _empty_estimate(
            "body_control", "Not enough elbow-angle samples. " + _SUBJECTIVITY_WARNING
        )
    jitter = float(np.std(np.diff(combined)))
    # Heuristic: ~0 degrees/frame jitter -> 10.0, >= 60 degrees/frame -> 0.0.
    value = float(max(0.0, 10.0 - (jitter / 6.0)))
    return FreestyleEvaluationEstimate(
        category="body_control",
        value=round(value, 3),
        confidence=_confidence_from_sample_size(int(combined.size), saturate_at=200),
        supporting_features={
            "elbow_angle_jitter_deg": jitter,
            "sample_count": float(combined.size),
        },
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        warning=_SUBJECTIVITY_WARNING,
    )


def estimate_music_choreography() -> FreestyleEvaluationEstimate:
    return _empty_estimate("music_choreography", _NO_AUDIO_WARNING)


def estimate_music_construction() -> FreestyleEvaluationEstimate:
    return _empty_estimate("music_construction", _NO_AUDIO_WARNING)


#: Categories with no estimator at all -- `showmanship` is intentionally
#: absent: Prompt D says "Keep showmanship manual by default," so there is
#: no `estimate_showmanship` function and `estimate_all` never returns it.
UNSUPPORTED_CATEGORIES: tuple[str, ...] = ("music_choreography", "music_construction")


def estimate_all(
    events: Sequence[AnalysisEventPrediction], features: FeatureSet | None
) -> dict[str, FreestyleEvaluationEstimate]:
    """Runs every automatic estimator and returns a category -> estimate
    map. Never includes a `"showmanship"` key."""
    return {
        "execution": estimate_execution(events),
        "control": estimate_control(events),
        "trick_diversity": estimate_trick_diversity(events),
        "space_use_emphasis": estimate_space_use_emphasis(features),
        "music_choreography": estimate_music_choreography(),
        "music_construction": estimate_music_construction(),
        "body_control": estimate_body_control(features),
    }
