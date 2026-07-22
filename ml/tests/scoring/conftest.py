"""Shared synthetic-data builders for the Prompt D `scoring/` package tests.

Every helper below builds a minimal, valid instance of its domain/dataset
type using only in-memory synthetic values (no fixture files, no production
data) -- mirroring the `_event()` helper already used by
`tests/test_scoring_engine.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from yoyovision_ml.domain import (
    AnalysisEvent,
    AnalysisEventPrediction,
    DeductionType,
    DifficultyBand,
    EventFamily,
    FeatureFrame,
    FeatureSet,
    MajorDeduction,
    Outcome,
    ReviewStatus,
    Source,
)
from yoyovision_ml.scoring.types import EventOverride, JudgeClick, JudgeFreestyleScore


def make_analysis_event(
    event_id: str = "evt-1",
    *,
    label: str = "mount_1",
    family: EventFamily = EventFamily.MOUNT,
    start_ms: int = 0,
    outcome: Outcome = Outcome.SUCCESS,
    band: DifficultyBand = DifficultyBand.BASIC,
    confidence: float = 0.9,
    source: Source = Source.MODEL,
    review_status: ReviewStatus = ReviewStatus.PENDING,
) -> AnalysisEvent:
    return AnalysisEvent(
        id=event_id,
        analysis_id="analysis-1",
        label=label,
        family=family,
        start_ms=start_ms,
        end_ms=start_ms + 500,
        confidence=confidence,
        outcome=outcome,
        difficulty_band=band,
        source=source,
        review_status=review_status,
        model_name="test-model" if source == Source.MODEL else None,
        model_version="0.0.0-test" if source == Source.MODEL else None,
    )


def make_event_prediction(
    *,
    label: str = "mount_1",
    family: EventFamily = EventFamily.MOUNT,
    start_ms: int = 0,
    outcome: Outcome = Outcome.SUCCESS,
    band: DifficultyBand = DifficultyBand.BASIC,
    confidence: float = 0.9,
) -> AnalysisEventPrediction:
    return AnalysisEventPrediction(
        label=label,
        family=family,
        start_ms=start_ms,
        end_ms=start_ms + 500,
        confidence=confidence,
        outcome=outcome,
        difficulty_band=band,
        model_name="test-model",
        model_version="0.0.0-test",
    )


def make_major_deduction(
    deduction_id: str = "ded-1",
    *,
    type_: DeductionType = DeductionType.YOYO_STOP,
    timestamp_ms: int = 0,
    quantity: int = 1,
    confidence: float = 0.9,
    source: Source = Source.MODEL,
    review_status: ReviewStatus = ReviewStatus.PENDING,
) -> MajorDeduction:
    return MajorDeduction(
        id=deduction_id,
        analysis_id="analysis-1",
        type=type_,
        timestamp_ms=timestamp_ms,
        quantity=quantity,
        points=0.0,
        confidence=confidence,
        source=source,
        review_status=review_status,
    )


def make_event_override(
    *,
    event_id: str = "evt-1",
    field_name: str = "outcome",
    original_value: str = "miss",
    overridden_value: str = "success",
    overridden_by: str = "reviewer-1",
    reason: str = "",
) -> EventOverride:
    return EventOverride(
        event_id=event_id,
        field_name=field_name,
        original_value=original_value,
        overridden_value=overridden_value,
        overridden_by=overridden_by,
        overridden_at=datetime(2026, 1, 1, tzinfo=UTC),
        reason=reason,
    )


def make_judge_click(
    *,
    click_id: str = "click-1",
    judge_id: str = "judge-a",
    timestamp_ms: int = 0,
    associated_label: str | None = None,
) -> JudgeClick:
    return JudgeClick(
        click_id=click_id,
        judge_id=judge_id,
        timestamp_ms=timestamp_ms,
        associated_label=associated_label,
    )


def make_judge_score(
    *,
    judge_id: str = "judge-a",
    execution: float | None = 7.0,
    control: float | None = 7.0,
    trick_diversity: float | None = 7.0,
    space_use_emphasis: float | None = 7.0,
    music_choreography: float | None = 7.0,
    music_construction: float | None = 7.0,
    body_control: float | None = 7.0,
    showmanship: float | None = 7.0,
) -> JudgeFreestyleScore:
    return JudgeFreestyleScore(
        judge_id=judge_id,
        execution=execution,
        control=control,
        trick_diversity=trick_diversity,
        space_use_emphasis=space_use_emphasis,
        music_choreography=music_choreography,
        music_construction=music_construction,
        body_control=body_control,
        showmanship=showmanship,
    )


def make_feature_set(
    *,
    frame_count: int = 10,
    frame_spacing_ms: int = 33,
    values_by_frame: list[dict[str, float]] | None = None,
    feature_names: tuple[str, ...] = (),
) -> FeatureSet:
    if values_by_frame is None:
        values_by_frame = [{} for _ in range(frame_count)]
    frames = tuple(
        FeatureFrame(frame_ms=i * frame_spacing_ms, values=values)
        for i, values in enumerate(values_by_frame)
    )
    return FeatureSet(frames=frames, feature_names=feature_names, fps=30.0)
