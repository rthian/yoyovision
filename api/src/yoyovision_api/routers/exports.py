"""JSON and CSV export endpoints for a completed analysis."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import select
from yoyovision_ml.exports import (
    export_analysis_json,
    export_deductions_csv,
    export_events_csv,
    sanitize_export_filename,
)

from yoyovision_api.db_models import (
    AnalysisEventORM,
    MajorDeductionORM,
    ScoreBreakdownORM,
    VideoAssetORM,
)
from yoyovision_api.deps import CurrentUser, DbSession, OwnedJob, SettingsDep, StorageDep
from yoyovision_api.services.dataset_export_service import build_dataset_record
from yoyovision_api.services.domain_mapping import (
    deduction_to_domain,
    event_to_domain,
    score_to_domain,
    video_to_domain,
)

router = APIRouter(prefix="/analyses/{analysis_id}/export", tags=["exports"])


async def _load_video(job: OwnedJob, session: DbSession) -> VideoAssetORM:
    result = await session.execute(select(VideoAssetORM).where(VideoAssetORM.id == job.video_id))
    video = result.scalar_one_or_none()
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")
    return video


@router.get("/report.json")
async def export_json(job: OwnedJob, session: DbSession, settings: SettingsDep) -> Response:
    video = await _load_video(job, session)
    events_result = await session.execute(
        select(AnalysisEventORM).where(AnalysisEventORM.analysis_id == job.id)
    )
    deductions_result = await session.execute(
        select(MajorDeductionORM).where(MajorDeductionORM.analysis_id == job.id)
    )
    score_result = await session.execute(
        select(ScoreBreakdownORM).where(ScoreBreakdownORM.analysis_id == job.id)
    )
    score_row = score_result.scalar_one_or_none()

    payload = export_analysis_json(
        video=video_to_domain(video),
        events=[event_to_domain(e) for e in events_result.scalars().all()],
        deductions=[deduction_to_domain(d) for d in deductions_result.scalars().all()],
        score=score_to_domain(score_row) if score_row is not None else None,
        pipeline_version=job.pipeline_version,
        ruleset_version=settings.ruleset_version,
    )
    filename = sanitize_export_filename(f"yoyovision-analysis-{job.id}", "json")
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/events.csv")
async def export_events(job: OwnedJob, session: DbSession) -> Response:
    result = await session.execute(
        select(AnalysisEventORM).where(AnalysisEventORM.analysis_id == job.id)
    )
    payload = export_events_csv([event_to_domain(e) for e in result.scalars().all()])
    filename = sanitize_export_filename(f"yoyovision-events-{job.id}", "csv")
    return Response(
        content=payload,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/deductions.csv")
async def export_deductions(job: OwnedJob, session: DbSession) -> Response:
    result = await session.execute(
        select(MajorDeductionORM).where(MajorDeductionORM.analysis_id == job.id)
    )
    payload = export_deductions_csv([deduction_to_domain(d) for d in result.scalars().all()])
    filename = sanitize_export_filename(f"yoyovision-deductions-{job.id}", "csv")
    return Response(
        content=payload,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/dataset-record.json")
async def export_dataset_record(
    job: OwnedJob,
    session: DbSession,
    current_user: CurrentUser,
    storage: StorageDep,
) -> Response:
    """Exports the reviewed analysis as a versioned `DatasetRecord` for training."""
    video = await _load_video(job, session)
    try:
        record = await build_dataset_record(session, job, video, current_user, storage)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    filename = sanitize_export_filename(f"yoyovision-dataset-record-{job.id}", "json")
    return Response(
        content=record.model_dump_json(indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
