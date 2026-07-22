"""Analysis job status, cancellation, and score retrieval/recomputation."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from yoyovision_ml.domain import JobStatus

from yoyovision_api.db_models import AnalysisJobORM, ScoreBreakdownORM
from yoyovision_api.deps import DbSession, OwnedJob, SettingsDep
from yoyovision_api.schemas import (
    AnalysisJobRead,
    ScoreBreakdownRead,
    ScoreLineItemsRead,
    ScorePreviewRead,
    TechnicalLineItemRead,
)
from yoyovision_api.services.scoring_service import (
    compute_score_line_items,
    compute_score_preview,
    recompute_score,
)

router = APIRouter(prefix="/analyses", tags=["analyses"])

_TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})


@router.get("/{analysis_id}", response_model=AnalysisJobRead)
async def get_analysis(job: OwnedJob) -> AnalysisJobORM:
    return job


@router.post("/{analysis_id}/cancel", response_model=AnalysisJobRead)
async def cancel_analysis(job: OwnedJob, session: DbSession) -> AnalysisJobORM:
    """Requests cooperative cancellation of a running (or still-queued)
    job (Prompt F). Sets `cancel_requested`, which the worker polls between
    pipeline stages via `CancellationToken` -- this does not guarantee
    immediate termination, only that the job stops at the next stage
    boundary rather than running to completion. A no-op (not an error) for
    jobs that have already reached a terminal status."""
    if job.status not in _TERMINAL_STATUSES:
        job.cancel_requested = True
        await session.commit()
    return job


@router.get("/{analysis_id}/score", response_model=ScoreBreakdownRead)
async def get_score(job: OwnedJob, session: DbSession, settings: SettingsDep) -> ScoreBreakdownORM:
    """Returns the current score, recomputing it from current DB state first
    so the returned breakdown always reflects the latest human edits."""
    breakdown = await recompute_score(session, job, settings.ruleset_version)
    await session.commit()
    return breakdown


@router.get("/{analysis_id}/score/line-items", response_model=ScoreLineItemsRead)
async def get_score_line_items(
    job: OwnedJob, session: DbSession, settings: SettingsDep
) -> ScoreLineItemsRead:
    """Returns per-event technical credit rows for the review UI."""
    technical_raw, items = await compute_score_line_items(session, job, settings.ruleset_version)
    return ScoreLineItemsRead(
        technical_raw=technical_raw,
        technical_line_items=[TechnicalLineItemRead.model_validate(item) for item in items],
    )


@router.get("/{analysis_id}/score/preview", response_model=ScorePreviewRead)
async def get_score_preview(
    job: OwnedJob,
    session: DbSession,
    settings: SettingsDep,
    up_to_ms: int,
) -> ScorePreviewRead:
    """Returns a playhead-gated score: tricks credit only after `end_ms`, and
    deductions apply only once `timestamp_ms` has passed."""
    if up_to_ms < 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="up_to_ms must be >= 0")

    breakdown, event_rows, completed_count = await compute_score_preview(
        session, job, settings.ruleset_version, up_to_ms
    )
    active_event_id = next(
        (
            row.id
            for row in event_rows
            if row.start_ms <= up_to_ms <= row.end_ms
        ),
        None,
    )
    return ScorePreviewRead(
        up_to_ms=up_to_ms,
        completed_event_count=completed_count,
        active_event_id=active_event_id,
        technical_raw=breakdown.technical_raw,
        technical_scaled=breakdown.technical_scaled,
        freestyle_evaluation_raw=breakdown.freestyle_evaluation_raw,
        freestyle_evaluation_scaled=breakdown.freestyle_evaluation_scaled,
        major_deductions=breakdown.major_deductions,
        final_score=breakdown.final_score,
        confidence=breakdown.confidence,
        ruleset_version=breakdown.ruleset_version,
        warnings=breakdown.warnings,
    )


@router.post("/{analysis_id}/score/recompute", response_model=ScoreBreakdownRead)
async def recompute_analysis_score(
    job: OwnedJob, session: DbSession, settings: SettingsDep
) -> ScoreBreakdownORM:
    """Explicit recompute endpoint (same effect as `GET .../score`, exposed
    separately so review-UI "Recalculate score" actions are self-documenting
    and show up distinctly in server logs/audit trails)."""
    breakdown = await recompute_score(session, job, settings.ruleset_version)
    await session.commit()
    return breakdown
