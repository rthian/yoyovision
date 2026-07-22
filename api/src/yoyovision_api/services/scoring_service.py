"""Recomputes and persists a `ScoreBreakdown` from the current, human-editable
DB state of an analysis (events + deductions + Freestyle Evaluation).

This is the single place that converts persisted ORM rows into the
`yoyovision_ml` domain/prediction types the deterministic scoring engine
consumes, so every code path that mutates events/deductions/evaluation can
call `recompute_score` and get an identical, fully audited result. Rejected
detections (`review_status == REJECTED`) are excluded from scoring -- a
human has judged them not to be real events -- but the rows themselves are
kept for the audit trail.

Deductions are additionally filtered through `yoyovision_ml.scoring_engine.
deduction_is_scorable` (Prompt D): a deduction whose ruleset rule sets
`requires_manual_confirmation=True` (currently only `dangerous_play_review`)
contributes zero score impact until a human explicitly sets its
`review_status` to `CONFIRMED` -- "Dangerous-play detection must never
automatically disqualify a player. It must create a review flag." A
freshly-detected flag (`PENDING`) is persisted and visible for review, but
`recompute_score` never lets it change `final_score` on its own.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from yoyovision_ml.domain import (
    AnalysisEventPrediction,
    DeductionPrediction,
    EvidenceRef,
    FreestyleEvaluation,
    ReviewStatus,
    TechnicalLineItem,
)
from yoyovision_ml.ruleset import Ruleset, default_ruleset, get_ruleset_by_version
from yoyovision_ml.scoring_engine import (
    DeterministicScoringEngine,
    deduction_is_scorable,
    technical_line_items,
)

from yoyovision_api.db_models import (
    AnalysisEventORM,
    AnalysisJobORM,
    FreestyleEvaluationORM,
    MajorDeductionORM,
    ScoreBreakdownORM,
)


def _event_to_prediction(event: AnalysisEventORM) -> AnalysisEventPrediction:
    return AnalysisEventPrediction(
        label=event.label,
        family=event.family,
        start_ms=event.start_ms,
        end_ms=event.end_ms,
        confidence=event.confidence,
        outcome=event.outcome,
        difficulty_band=event.difficulty_band,
        model_name=event.model_name or "human",
        model_version=event.model_version or "n/a",
        evidence=(EvidenceRef(frame_ms=event.start_ms, note="see AnalysisEvent.evidence_json"),),
    )


def _deduction_to_prediction(deduction: MajorDeductionORM) -> DeductionPrediction:
    return DeductionPrediction(
        type=deduction.type,
        timestamp_ms=deduction.timestamp_ms,
        quantity=deduction.quantity,
        confidence=deduction.confidence,
        model_name="human" if deduction.source.value == "human" else "mock-temporal-event-detector",
        model_version="n/a",
        points=deduction.points,
    )


def _evaluation_to_domain(evaluation: FreestyleEvaluationORM | None) -> FreestyleEvaluation | None:
    if evaluation is None:
        return None
    return FreestyleEvaluation(
        execution=evaluation.execution,
        control=evaluation.control,
        trick_diversity=evaluation.trick_diversity,
        space_use_emphasis=evaluation.space_use_emphasis,
        music_choreography=evaluation.music_choreography,
        music_construction=evaluation.music_construction,
        body_control=evaluation.body_control,
        showmanship=evaluation.showmanship,
        source=evaluation.source,
        notes=evaluation.notes,
    )


def resolve_ruleset(ruleset_version: str) -> Ruleset:
    return get_ruleset_by_version(ruleset_version) or default_ruleset()


async def recompute_score(
    session: AsyncSession, job: AnalysisJobORM, ruleset_version: str
) -> ScoreBreakdownORM:
    """Recomputes the score for `job` from its current DB state and upserts
    the single `ScoreBreakdownORM` row for that analysis. Callers are
    responsible for committing the session."""
    events_result = await session.execute(
        select(AnalysisEventORM).where(AnalysisEventORM.analysis_id == job.id)
    )
    deductions_result = await session.execute(
        select(MajorDeductionORM).where(MajorDeductionORM.analysis_id == job.id)
    )
    evaluation_result = await session.execute(
        select(FreestyleEvaluationORM).where(FreestyleEvaluationORM.analysis_id == job.id)
    )

    events = [
        _event_to_prediction(e)
        for e in events_result.scalars().all()
        if e.review_status != ReviewStatus.REJECTED
    ]
    ruleset = resolve_ruleset(ruleset_version)
    deductions = [
        _deduction_to_prediction(d)
        for d in deductions_result.scalars().all()
        if deduction_is_scorable(d.type, d.review_status, ruleset)
    ]
    evaluation = _evaluation_to_domain(evaluation_result.scalar_one_or_none())
    breakdown = DeterministicScoringEngine().calculate(
        events=events, deductions=deductions, freestyle_evaluation=evaluation, ruleset=ruleset
    )

    existing_result = await session.execute(
        select(ScoreBreakdownORM).where(ScoreBreakdownORM.analysis_id == job.id)
    )
    existing = existing_result.scalar_one_or_none()
    if existing is None:
        existing = ScoreBreakdownORM(analysis_id=job.id)
        session.add(existing)

    existing.technical_raw = breakdown.technical_raw
    existing.technical_scaled = breakdown.technical_scaled
    existing.freestyle_evaluation_raw = breakdown.freestyle_evaluation_raw
    existing.freestyle_evaluation_scaled = breakdown.freestyle_evaluation_scaled
    existing.major_deductions = breakdown.major_deductions
    existing.final_score = breakdown.final_score
    existing.confidence = breakdown.confidence
    existing.ruleset_version = breakdown.ruleset_version
    existing.warnings = breakdown.warnings

    await session.flush()
    return existing


async def compute_score_line_items(
    session: AsyncSession, job: AnalysisJobORM, ruleset_version: str
) -> tuple[float, list[TechnicalLineItem]]:
    """Returns per-event technical credit rows for the review UI, using the
    same filtering rules as `recompute_score`."""
    events_result = await session.execute(
        select(AnalysisEventORM).where(AnalysisEventORM.analysis_id == job.id)
    )
    event_rows = [
        row for row in events_result.scalars().all() if row.review_status != ReviewStatus.REJECTED
    ]
    ruleset = resolve_ruleset(ruleset_version)
    predictions = [_event_to_prediction(row) for row in event_rows]
    event_ids = [row.id for row in event_rows]
    technical_raw, _, items = technical_line_items(
        predictions, ruleset, event_ids=event_ids
    )
    return technical_raw, items
