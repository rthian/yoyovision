"""Build a `DatasetRecord` from a reviewed analysis for model training."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from yoyovision_ml.dataset.schema import (
    AnnotationProvenance,
    DatasetRecord,
    DatasetVideo,
    DeductionAnnotation,
    FreestyleEvaluationAnnotation,
    TrickEventAnnotation,
)
from yoyovision_ml.domain import AnalysisReviewState, ReviewStatus
from yoyovision_ml.interfaces import StoragePort

from yoyovision_api.db_models import (
    AnalysisEventORM,
    AnalysisJobORM,
    FreestyleEvaluationORM,
    MajorDeductionORM,
    User,
    VideoAssetORM,
)

DATASET_ONTOLOGY_VERSION = "dataset-ontology-v1"
DATASET_SCHEMA_VERSION = "1.0.0"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _provenance_for(
    *,
    annotator_id: str,
    source,
    annotated_at: datetime,
    is_adjudicated: bool,
    adjudicated_by: str | None,
) -> AnnotationProvenance:
    return AnnotationProvenance(
        annotator_id=annotator_id,
        source=source,
        annotated_at=annotated_at,
        tool="yoyovision-review-ui",
        tool_version=None,
        is_adjudicated=is_adjudicated,
        adjudicated_by=adjudicated_by,
    )


async def build_dataset_record(
    session: AsyncSession,
    job: AnalysisJobORM,
    video: VideoAssetORM,
    reviewer: User,
    storage: StoragePort,
) -> DatasetRecord:
    """Exports non-rejected review state as a versioned `DatasetRecord`."""
    events_result = await session.execute(
        select(AnalysisEventORM)
        .where(AnalysisEventORM.analysis_id == job.id)
        .order_by(AnalysisEventORM.start_ms)
    )
    deductions_result = await session.execute(
        select(MajorDeductionORM)
        .where(MajorDeductionORM.analysis_id == job.id)
        .order_by(MajorDeductionORM.timestamp_ms)
    )
    evaluation_result = await session.execute(
        select(FreestyleEvaluationORM).where(FreestyleEvaluationORM.analysis_id == job.id)
    )

    is_adjudicated = job.review_state == AnalysisReviewState.SUBMITTED
    adjudicated_by = reviewer.email if is_adjudicated else None
    annotator_id = reviewer.email

    try:
        checksum_sha256 = _sha256_hex(storage.get(video.storage_key))
    except Exception:
        checksum_sha256 = _sha256_hex(video.id.encode("utf-8"))

    duration_ms = video.duration_ms or 0
    if duration_ms <= 0:
        raise ValueError("Video duration is required to export a dataset record.")

    routine_note_parts: list[str] = []
    if job.routine_start_ms is not None:
        routine_note_parts.append(f"routine_start_ms={job.routine_start_ms}")
    if job.routine_end_ms is not None:
        routine_note_parts.append(f"routine_end_ms={job.routine_end_ms}")
    video_notes = "Exported from YoYoVision analysis review."
    if routine_note_parts:
        video_notes = f"{video_notes} {'; '.join(routine_note_parts)}."

    dataset_video = DatasetVideo(
        video_id=video.id,
        player_id=f"player-{video.owner_id[:8]}",
        division="1A",
        relative_path=video.storage_key,
        checksum_sha256=checksum_sha256,
        duration_ms=duration_ms,
        width=video.width or 1280,
        height=video.height or 720,
        source_fps=video.fps or 30.0,
        notes=video_notes,
    )

    trick_events: list[TrickEventAnnotation] = []
    for event in events_result.scalars().all():
        if event.review_status == ReviewStatus.REJECTED:
            continue
        trick_events.append(
            TrickEventAnnotation(
                event_id=event.id,
                label=event.label,
                family=event.family,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
                outcome=event.outcome,
                difficulty_band=event.difficulty_band,
                confidence=event.confidence,
                provenance=_provenance_for(
                    annotator_id=annotator_id,
                    source=event.source,
                    annotated_at=event.updated_at or event.created_at,
                    is_adjudicated=is_adjudicated,
                    adjudicated_by=adjudicated_by,
                ),
            )
        )

    deductions: list[DeductionAnnotation] = []
    for deduction in deductions_result.scalars().all():
        if deduction.review_status == ReviewStatus.REJECTED:
            continue
        deductions.append(
            DeductionAnnotation(
                deduction_id=deduction.id,
                type=deduction.type,
                timestamp_ms=deduction.timestamp_ms,
                quantity=deduction.quantity,
                confidence=deduction.confidence,
                provenance=_provenance_for(
                    annotator_id=annotator_id,
                    source=deduction.source,
                    annotated_at=datetime.now(UTC),
                    is_adjudicated=is_adjudicated,
                    adjudicated_by=adjudicated_by,
                ),
            )
        )

    freestyle_evaluations: list[FreestyleEvaluationAnnotation] = []
    evaluation = evaluation_result.scalar_one_or_none()
    if evaluation is not None:
        freestyle_evaluations.append(
            FreestyleEvaluationAnnotation(
                judge_id=annotator_id,
                execution=evaluation.execution,
                control=evaluation.control,
                trick_diversity=evaluation.trick_diversity,
                space_use_emphasis=evaluation.space_use_emphasis,
                music_choreography=evaluation.music_choreography,
                music_construction=evaluation.music_construction,
                body_control=evaluation.body_control,
                showmanship=evaluation.showmanship,
                provenance=_provenance_for(
                    annotator_id=annotator_id,
                    source=evaluation.source,
                    annotated_at=datetime.now(UTC),
                    is_adjudicated=is_adjudicated,
                    adjudicated_by=adjudicated_by,
                ),
                notes=evaluation.notes,
            )
        )

    record_id = f"{job.id}__{annotator_id.replace('@', '_at_')}"
    return DatasetRecord(
        record_id=record_id,
        video=dataset_video,
        annotator_id=annotator_id,
        is_adjudicated=is_adjudicated,
        schema_version=DATASET_SCHEMA_VERSION,
        ontology_version=DATASET_ONTOLOGY_VERSION,
        trick_events=trick_events,
        deductions=deductions,
        freestyle_evaluations=freestyle_evaluations,
    )
