"""Analysis job status, cancellation, and score retrieval/recomputation."""

from __future__ import annotations

from fastapi import APIRouter
from yoyovision_ml.domain import JobStatus

from yoyovision_api.db_models import AnalysisJobORM, ScoreBreakdownORM
from yoyovision_api.deps import DbSession, OwnedJob, SettingsDep
from yoyovision_api.schemas import (
    AnalysisJobRead,
    ScoreBreakdownRead,
    ScoreLineItemsRead,
    TechnicalLineItemRead,
)
from yoyovision_api.services.scoring_service import compute_score_line_items, recompute_score

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
