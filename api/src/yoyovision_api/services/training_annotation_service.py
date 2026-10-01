"""Export human trick labels as a versioned training dataset record."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from yoyovision_ml.dataset.schema import (
    AnnotationProvenance,
    DatasetRecord,
    DatasetVideo,
    TrickEventAnnotation,
)
from yoyovision_ml.domain import DifficultyBand, EventFamily, Source
from yoyovision_ml.interfaces import StoragePort

from yoyovision_api.db_models import TrainingAnnotationORM, User, VideoAssetORM


async def build_training_record(
    session: AsyncSession,
    video: VideoAssetORM,
    annotator: User,
    storage: StoragePort,
) -> DatasetRecord:
    result = await session.execute(
        select(TrainingAnnotationORM)
        .where(TrainingAnnotationORM.video_id == video.id)
        .order_by(TrainingAnnotationORM.start_ms)
    )
    annotations = list(result.scalars().all())
    if not annotations:
        raise ValueError("Add at least one trick annotation before exporting training data.")
    if not video.duration_ms or video.duration_ms <= 0:
        raise ValueError("Video duration is required to export training data.")
    if not video.player_id:
        raise ValueError("A performer ID is required to export training data.")
    if video.rights_confirmed_at is None:
        raise ValueError("Rights confirmation is required to export training data.")

    data = storage.get(video.storage_key)
    dataset_video = DatasetVideo(
        video_id=video.id,
        player_id=video.player_id,
        division=video.division,
        relative_path=video.storage_key,
        checksum_sha256=hashlib.sha256(data).hexdigest(),
        duration_ms=video.duration_ms,
        width=video.width or 1280,
        height=video.height or 720,
        source_fps=video.fps or 30.0,
        consent_reference=(
            f"rights-confirmed:{video.rights_confirmed_at.isoformat()}"
            if video.rights_confirmed_at
            else None
        ),
        source_url=video.source_url,
        rights_confirmed=video.rights_confirmed_at is not None,
        notes="Exported from the YoYoVision human training-annotation workflow.",
    )
    trick_events = [
        TrickEventAnnotation(
            event_id=item.id,
            label=item.label,
            family=EventFamily.UNKNOWN_TECHNICAL_ELEMENT,
            element_type=item.element_type,
            start_ms=item.start_ms,
            end_ms=item.end_ms,
            outcome=item.outcome,
            difficulty_band=DifficultyBand.UNKNOWN,
            technical_credit=item.technical_credit,
            confidence=1.0,
            provenance=AnnotationProvenance(
                annotator_id=annotator.email,
                source=Source.HUMAN,
                annotated_at=item.updated_at or item.created_at or datetime.now(UTC),
                tool="yoyovision-training-annotation-ui",
            ),
            notes=item.notes,
        )
        for item in annotations
    ]
    return DatasetRecord(
        record_id=f"{video.id}__{annotator.id}",
        video=dataset_video,
        annotator_id=annotator.email,
        ontology_version=f"{video.division.value.lower()}-draft-v1",
        trick_events=trick_events,
    )
