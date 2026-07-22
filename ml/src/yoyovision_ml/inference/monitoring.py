"""Monitoring signals: class drift, confidence drift, failed-track rate.

Prompt F: "Add monitoring for class drift, confidence drift and
failed-track rate." No metrics dashboard exists in this repo, so these
signals are computed deterministically per job and returned on
`PipelineResult`/logged structurally -- a real deployment would ship them to
whatever metrics backend it has; the computation itself is what this module
owns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from yoyovision_ml.domain import AnalysisEventPrediction, Track

_LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass(slots=True, frozen=True)
class ReferenceBaseline:
    """A reference distribution to compare a job's signals against, e.g.
    computed once from a validation set. `class_frequencies` should sum to
    ~1.0 across event families."""

    class_frequencies: dict[str, float]
    avg_confidence: float


@dataclass(slots=True, frozen=True)
class MonitoringSignals:
    class_counts: dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    low_confidence_rate: float = 0.0
    failed_track_rate: float = 0.0
    class_drift_score: float | None = None
    confidence_drift_score: float | None = None


def _class_drift_score(
    class_counts: dict[str, int], reference: dict[str, float]
) -> float:
    """Total-variation distance (half the L1 distance) between this job's
    normalized class distribution and `reference`. `0.0` means identical
    distributions; `1.0` means completely disjoint. Simple and
    interpretable on purpose -- this is a monitoring signal, not a
    statistical test."""
    total = sum(class_counts.values())
    if total == 0:
        return 0.0
    observed = {label: count / total for label, count in class_counts.items()}
    labels = set(observed) | set(reference)
    return sum(abs(observed.get(label, 0.0) - reference.get(label, 0.0)) for label in labels) / 2.0


def compute_monitoring_signals(
    events: list[AnalysisEventPrediction],
    tracks: list[Track],
    *,
    low_confidence_threshold: float = _LOW_CONFIDENCE_THRESHOLD,
    reference: ReferenceBaseline | None = None,
) -> MonitoringSignals:
    """Computes per-job monitoring signals from this run's predicted events
    and yo-yo tracks.

    - `class_counts` / `class_drift_score`: distribution of predicted event
      families vs. `reference.class_frequencies`, if provided.
    - `avg_confidence` / `confidence_drift_score`: mean event confidence vs.
      `reference.avg_confidence`.
    - `failed_track_rate`: fraction of tracked frames that were
      interpolated or not fully visible -- a proxy for "the tracker lost the
      yo-yo" that does not require ground truth.
    """
    class_counts: dict[str, int] = {}
    for event in events:
        class_counts[event.family.value] = class_counts.get(event.family.value, 0) + 1

    confidences = [event.confidence for event in events]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    low_confidence_rate = (
        sum(1 for c in confidences if c < low_confidence_threshold) / len(confidences)
        if confidences
        else 0.0
    )

    failed_tracks = sum(
        1 for track in tracks if track.interpolated or track.confidence < low_confidence_threshold
    )
    failed_track_rate = failed_tracks / len(tracks) if tracks else 0.0

    class_drift_score = (
        _class_drift_score(class_counts, reference.class_frequencies) if reference else None
    )
    confidence_drift_score = (
        abs(avg_confidence - reference.avg_confidence) if reference else None
    )

    return MonitoringSignals(
        class_counts=class_counts,
        avg_confidence=avg_confidence,
        low_confidence_rate=low_confidence_rate,
        failed_track_rate=failed_track_rate,
        class_drift_score=class_drift_score,
        confidence_drift_score=confidence_drift_score,
    )
