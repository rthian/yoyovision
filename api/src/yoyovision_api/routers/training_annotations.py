"""Human-authored trick annotations used to build real training corpora."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import select
from yoyovision_ml.dataset.schema import DatasetRecord

from yoyovision_api.db_models import TrainingAnnotationORM
from yoyovision_api.deps import CurrentUser, DbSession, OwnedVideo, StorageDep
from yoyovision_api.schemas import (
    TrainingAnnotationCreate,
    TrainingAnnotationRead,
    TrainingAnnotationUpdate,
)
from yoyovision_api.services.training_annotation_service import build_training_record

router = APIRouter(prefix="/videos/{video_id}", tags=["training-annotations"])


def _validate_window(video: OwnedVideo, start_ms: int, end_ms: int) -> None:
    if end_ms <= start_ms:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="end_ms must be greater than start_ms.",
        )
    if video.duration_ms is not None and end_ms > video.duration_ms:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Annotation cannot end after the video duration.",
        )


async def _get_annotation(
    session: DbSession, video: OwnedVideo, annotation_id: str
) -> TrainingAnnotationORM:
    result = await session.execute(
        select(TrainingAnnotationORM).where(
            TrainingAnnotationORM.id == annotation_id,
            TrainingAnnotationORM.video_id == video.id,
        )
    )
    annotation = result.scalar_one_or_none()
    if annotation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotation not found.")
    return annotation


@router.get("/annotations", response_model=list[TrainingAnnotationRead])
async def list_annotations(
    video: OwnedVideo, session: DbSession
) -> list[TrainingAnnotationORM]:
    result = await session.execute(
        select(TrainingAnnotationORM)
        .where(TrainingAnnotationORM.video_id == video.id)
        .order_by(TrainingAnnotationORM.start_ms)
    )
    return list(result.scalars().all())


@router.post(
    "/annotations", response_model=TrainingAnnotationRead, status_code=status.HTTP_201_CREATED
)
async def create_annotation(
    video: OwnedVideo,
    payload: TrainingAnnotationCreate,
    session: DbSession,
    current_user: CurrentUser,
) -> TrainingAnnotationORM:
    _validate_window(video, payload.start_ms, payload.end_ms)
    annotation = TrainingAnnotationORM(
        video_id=video.id,
        created_by=current_user.id,
        division=video.division,
        **payload.model_dump(),
    )
    session.add(annotation)
    await session.commit()
    await session.refresh(annotation)
    return annotation


@router.patch("/annotations/{annotation_id}", response_model=TrainingAnnotationRead)
async def update_annotation(
    video: OwnedVideo,
    annotation_id: str,
    payload: TrainingAnnotationUpdate,
    session: DbSession,
) -> TrainingAnnotationORM:
    annotation = await _get_annotation(session, video, annotation_id)
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    start_ms = int(changes.get("start_ms", annotation.start_ms))
    end_ms = int(changes.get("end_ms", annotation.end_ms))
    _validate_window(video, start_ms, end_ms)
    for field_name, value in changes.items():
        setattr(annotation, field_name, value)
    await session.commit()
    await session.refresh(annotation)
    return annotation


@router.delete("/annotations/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_annotation(
    video: OwnedVideo, annotation_id: str, session: DbSession
) -> Response:
    annotation = await _get_annotation(session, video, annotation_id)
    await session.delete(annotation)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/training-record", response_model=DatasetRecord)
async def export_training_record(
    video: OwnedVideo,
    session: DbSession,
    current_user: CurrentUser,
    storage: StorageDep,
) -> DatasetRecord:
    try:
        return await build_training_record(session, video, current_user, storage)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
