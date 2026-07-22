"""Dataset statistics, class-distribution reports, and annotator agreement.

Kept deliberately simple for this first cut: whole-corpus counts/durations
(requirement 10) and a lightweight pairwise annotator-agreement summary
(requirement 6's "reviewer agreement" support). Full calibration statistics
(MAE, Spearman/Pearson/ICC against expert judges) are explicitly out of
scope here -- see Prompt D.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from yoyovision_ml.dataset.schema import DatasetRecord, SplitName
from yoyovision_ml.domain import DifficultyBand, EventFamily, Outcome


@dataclass(slots=True)
class DatasetStatistics:
    video_count: int
    player_count: int
    record_count: int
    total_duration_ms: int
    event_count: int
    deduction_count: int
    events_by_family: Counter[EventFamily] = field(default_factory=Counter)
    events_by_outcome: Counter[Outcome] = field(default_factory=Counter)
    events_by_difficulty_band: Counter[DifficultyBand] = field(default_factory=Counter)
    videos_by_split: Counter[SplitName] = field(default_factory=Counter)


def compute_dataset_statistics(
    records: list[DatasetRecord],
    video_splits: dict[str, SplitName] | None = None,
) -> DatasetStatistics:
    video_ids = {r.video.video_id for r in records}
    player_ids = {r.video.player_id for r in records}
    # Sum duration once per distinct video, not once per annotation record
    # (a video can have multiple un-adjudicated records).
    duration_by_video = {r.video.video_id: r.video.duration_ms for r in records}

    events_by_family: Counter[EventFamily] = Counter()
    events_by_outcome: Counter[Outcome] = Counter()
    events_by_band: Counter[DifficultyBand] = Counter()
    event_count = 0
    deduction_count = 0

    for record in records:
        for event in record.trick_events:
            event_count += 1
            events_by_family[event.family] += 1
            events_by_outcome[event.outcome] += 1
            events_by_band[event.difficulty_band] += 1
        deduction_count += len(record.deductions)

    videos_by_split: Counter[SplitName] = Counter()
    if video_splits:
        for video_id in video_ids:
            split = video_splits.get(video_id)
            if split is not None:
                videos_by_split[split] += 1

    return DatasetStatistics(
        video_count=len(video_ids),
        player_count=len(player_ids),
        record_count=len(records),
        total_duration_ms=sum(duration_by_video.values()),
        event_count=event_count,
        deduction_count=deduction_count,
        events_by_family=events_by_family,
        events_by_outcome=events_by_outcome,
        events_by_difficulty_band=events_by_band,
        videos_by_split=videos_by_split,
    )


@dataclass(slots=True, frozen=True)
class AnnotatorAgreement:
    video_id: str
    annotator_a: str
    annotator_b: str
    #: events matched between the two annotators via temporal overlap + same family
    matched_event_count: int
    #: events either annotator recorded that the other did not match
    unmatched_event_count: int
    #: matched_event_count / (matched + unmatched); 1.0 == perfect overlap-based agreement
    agreement_ratio: float
    #: of the matched pairs, how many also agreed on outcome (success/miss/uncertain)
    outcome_agreement_ratio: float | None


def _events_overlap(a_start: int, a_end: int, b_start: int, b_end: int, min_iou: float) -> bool:
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    if intersection <= 0:
        return False
    union = max(a_end, b_end) - min(a_start, b_start)
    iou = intersection / union if union > 0 else 0.0
    return iou >= min_iou


def compute_annotator_agreement(
    records: list[DatasetRecord], *, min_iou: float = 0.5
) -> list[AnnotatorAgreement]:
    """Pairwise agreement between every two non-adjudicated annotation
    passes over the same video, matched by temporal IoU + family.

    This is a coarse, dependency-free agreement signal (not Cohen's kappa or
    an IoU-weighted mAP) intended to flag videos that clearly need
    adjudication attention, not to be a publishable inter-annotator
    reliability statistic.
    """
    by_video: dict[str, list[DatasetRecord]] = {}
    for record in records:
        by_video.setdefault(record.video.video_id, []).append(record)

    results: list[AnnotatorAgreement] = []
    for video_id, video_records in by_video.items():
        passes = [r for r in video_records if not r.is_adjudicated]
        for i in range(len(passes)):
            for j in range(i + 1, len(passes)):
                results.append(_pairwise_agreement(video_id, passes[i], passes[j], min_iou))
    return results


def _pairwise_agreement(
    video_id: str, record_a: DatasetRecord, record_b: DatasetRecord, min_iou: float
) -> AnnotatorAgreement:
    events_b = list(record_b.trick_events)
    matched_b_indices: set[int] = set()
    matched_pairs = []

    for event_a in record_a.trick_events:
        for idx, event_b in enumerate(events_b):
            if idx in matched_b_indices:
                continue
            if event_a.family != event_b.family:
                continue
            if _events_overlap(
                event_a.start_ms, event_a.end_ms, event_b.start_ms, event_b.end_ms, min_iou
            ):
                matched_b_indices.add(idx)
                matched_pairs.append((event_a, event_b))
                break

    matched_count = len(matched_pairs)
    total_events = len(record_a.trick_events) + len(events_b)
    unmatched_count = total_events - 2 * matched_count

    agreement_ratio = matched_count / (matched_count + unmatched_count) if total_events > 0 else 1.0

    outcome_agreement_ratio: float | None = None
    if matched_pairs:
        agreeing_outcomes = sum(1 for a, b in matched_pairs if a.outcome == b.outcome)
        outcome_agreement_ratio = agreeing_outcomes / len(matched_pairs)

    return AnnotatorAgreement(
        video_id=video_id,
        annotator_a=record_a.annotator_id,
        annotator_b=record_b.annotator_id,
        matched_event_count=matched_count,
        unmatched_event_count=unmatched_count,
        agreement_ratio=round(agreement_ratio, 4),
        outcome_agreement_ratio=(
            round(outcome_agreement_ratio, 4) if outcome_agreement_ratio is not None else None
        ),
    )
