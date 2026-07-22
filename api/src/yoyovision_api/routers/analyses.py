"""Analysis job status, cancellation, and score retrieval/recomputation."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from yoyovision_ml.domain import AnalysisReviewState, JobStatus

from yoyovision_api.db_models import AnalysisJobORM, ScoreBreakdownORM, VideoAssetORM
from yoyovision_api.deps import CurrentUser, DbSession, OwnedJob, SettingsDep
from yoyovision_api.schemas import (
    AnalysisJobRead,
    RoutineWindowUpdate,
    ScoreBreakdownRead,
    ScoreLineItemsRead,
    ScorePreviewRead,
    TechnicalLineItemRead,
)
from yoyovision_api.services.review_guard import ensure_analysis_editable, ensure_analysis_submittable
from yoyovision_api.services.scoring_service import (
    compute_score_line_items,
    compute_score_preview,
    recompute_score,
    resolve_routine_window,
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


@router.patch("/{analysis_id}/routine-window", response_model=AnalysisJobRead)
async def update_routine_window(
    job: OwnedJob,
    payload: RoutineWindowUpdate,
    session: DbSession,
    settings: SettingsDep,
) -> AnalysisJobORM:
    """Sets the judged routine span (measure start through music stop) within
    the uploaded clip. Scoring and live playback respect these bounds."""
    ensure_analysis_editable(job)
    video = (
        await session.execute(select(VideoAssetORM).where(VideoAssetORM.id == job.video_id))
    ).scalar_one_or_none()
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")

    duration_ms = video.duration_ms or 0
    if payload.routine_start_ms is not None:
        job.routine_start_ms = payload.routine_start_ms
    if payload.routine_end_ms is not None:
        job.routine_end_ms = payload.routine_end_ms

    routine_start_ms, routine_end_ms = resolve_routine_window(job, duration_ms)
    if duration_ms > 0 and routine_end_ms > duration_ms:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="routine_end_ms cannot exceed video duration.",
        )
    if routine_start_ms < 0 or routine_end_ms <= routine_start_ms:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="routine_end_ms must be greater than routine_start_ms.",
        )

    await recompute_score(session, job, settings.ruleset_version)
    await session.commit()
    await session.refresh(job)
    return job


@router.post("/{analysis_id}/submit", response_model=AnalysisJobRead)
async def submit_analysis(
    job: OwnedJob,
    session: DbSession,
    current_user: CurrentUser,
) -> AnalysisJobORM:
    """Locks the analysis for editing after review is complete."""
    ensure_analysis_submittable(job)
    job.review_state = AnalysisReviewState.SUBMITTED
    job.submitted_at = datetime.now(UTC)
    job.submitted_by = current_user.id
    await session.commit()
    await session.refresh(job)
    return job


@router.post("/{analysis_id}/reopen", response_model=AnalysisJobRead)
async def reopen_analysis(job: OwnedJob, session: DbSession) -> AnalysisJobORM:
    """Returns a submitted analysis to draft so edits can resume."""
    if job.review_state != AnalysisReviewState.SUBMITTED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only submitted analyses can be reopened.",
        )
    job.review_state = AnalysisReviewState.DRAFT
    job.submitted_at = None
    job.submitted_by = None
    await session.commit()
    await session.refresh(job)
    return job
