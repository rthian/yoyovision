"""Persist reviewed analyses into the on-disk training corpus."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from yoyovision_ml.dataset.corpus import append_record_to_corpus
from yoyovision_ml.domain import AnalysisReviewState
from yoyovision_ml.interfaces import StoragePort

from yoyovision_api.config import Settings
from yoyovision_api.db_models import AnalysisJobORM, User, VideoAssetORM
from yoyovision_api.services.dataset_export_service import build_dataset_record


class CorpusNotConfiguredError(RuntimeError):
    """Raised when `dataset_corpus_root` is unset."""


class CorpusExportError(ValueError):
    """Raised when an analysis is not eligible for corpus export."""


async def append_analysis_to_corpus(
    session: AsyncSession,
    settings: Settings,
    job: AnalysisJobORM,
    video: VideoAssetORM,
    reviewer: User,
    storage: StoragePort,
) -> tuple[Path, str]:
    """Exports a submitted analysis into the configured training corpus directory."""
    if not settings.dataset_corpus_root:
        raise CorpusNotConfiguredError(
            "DATASET_CORPUS_ROOT is not configured on the API service."
        )
    if job.review_state != AnalysisReviewState.SUBMITTED:
        raise CorpusExportError("Only submitted analyses can be added to the training corpus.")

    record = await build_dataset_record(session, job, video, reviewer, storage)
    video_bytes = storage.get(video.storage_key)
    corpus_dir = Path(settings.dataset_corpus_root)
    record_path = append_record_to_corpus(
        corpus_dir,
        record,
        video_bytes,
        video_filename=video.original_filename,
    )
    relative_record_path = str(record_path.relative_to(corpus_dir))
    return record_path, relative_record_path
