"""Analysis-job creation and dispatch to the Celery pipeline task."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from yoyovision_ml.domain import JobStatus, PipelineStage

from yoyovision_api.celery_client import enqueue_analysis_pipeline
from yoyovision_api.config import Settings
from yoyovision_api.db_models import AnalysisJobORM, VideoAssetORM


async def create_and_dispatch_analysis_job(
    session: AsyncSession, settings: Settings, video: VideoAssetORM, is_shadow: bool = False
) -> AnalysisJobORM:
    """Creates a `pending` `AnalysisJobORM` row and enqueues the worker task
    that will run the (currently mock-adapter) pipeline for it. Per product
    principle #10 (offline asynchronous analysis), the API never runs the
    pipeline in-process.

    `is_shadow=True` (Prompt F "shadow mode") still runs the full pipeline
    and persists real events/deductions/score, but flags the job so clients
    know not to treat it as the video's official/canonical result -- useful
    for trying a new model version against real traffic without it counting.
    """
    job = AnalysisJobORM(
        video_id=video.id,
        status=JobStatus.PENDING,
        progress=0.0,
        current_stage=PipelineStage.QUEUED,
        pipeline_version=settings.pipeline_version,
        is_shadow=is_shadow,
    )
    session.add(job)
    await session.flush()

    enqueue_analysis_pipeline(settings, job.id)
    return job
