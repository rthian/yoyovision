"""Canonical trick library and cross-view tutorial/stage examples."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import selectinload
from yoyovision_ml.domain import Division

from yoyovision_api.db_models import TrickCatalogORM, TrickExampleORM, VideoAssetORM
from yoyovision_api.deps import CurrentUser, DbSession
from yoyovision_api.schemas import (
    TrickCatalogCreate,
    TrickCatalogRead,
    TrickCatalogUpdate,
    TrickExampleCreate,
    TrickExampleRead,
    TrickExampleUpdate,
)

router = APIRouter(prefix="/tricks", tags=["trick-library"])


def _example_read(example: TrickExampleORM) -> TrickExampleRead:
    return TrickExampleRead(
        id=example.id,
        trick_id=example.trick_id,
        video_id=example.video_id,
        start_ms=example.start_ms,
        end_ms=example.end_ms,
        view_type=example.view_type,  # type: ignore[arg-type]
        camera_angle=example.camera_angle,
        playback_speed=example.playback_speed,
        notes=example.notes,
        is_primary=example.is_primary,
        original_filename=example.video.original_filename,
        source_type=example.video.source_type,
        source_url=example.video.source_url,
        created_at=example.created_at,
        updated_at=example.updated_at,
    )


def _catalog_read(trick: TrickCatalogORM) -> TrickCatalogRead:
    return TrickCatalogRead(
        id=trick.id,
        owner_id=trick.owner_id,
        division=trick.division,
        name=trick.name,
        aliases=list(trick.aliases),
        description=trick.description,
        examples=[_example_read(example) for example in trick.examples],
        created_at=trick.created_at,
        updated_at=trick.updated_at,
    )


async def _get_owned_trick(
    session: DbSession, current_user: CurrentUser, trick_id: str
) -> TrickCatalogORM:
    result = await session.execute(
        select(TrickCatalogORM)
        .options(selectinload(TrickCatalogORM.examples).selectinload(TrickExampleORM.video))
        .where(TrickCatalogORM.id == trick_id, TrickCatalogORM.owner_id == current_user.id)
    )
    trick = result.scalar_one_or_none()
    if trick is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trick not found.")
    return trick


async def _assert_unique_name(
    session: DbSession,
    owner_id: str,
    division: Division,
    name: str,
    *,
    exclude_id: str | None = None,
) -> None:
    query = select(TrickCatalogORM.id).where(
        TrickCatalogORM.owner_id == owner_id,
        TrickCatalogORM.division == division,
        func.lower(TrickCatalogORM.name) == name.lower(),
    )
    if exclude_id:
        query = query.where(TrickCatalogORM.id != exclude_id)
    if (await session.execute(query)).scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A trick with this name already exists in the division.",
        )


def _validate_example_window(video: VideoAssetORM, start_ms: int, end_ms: int) -> None:
    if end_ms <= start_ms:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="end_ms must be greater than start_ms.",
        )
    if video.duration_ms is not None and end_ms > video.duration_ms:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Example cannot end after the video duration.",
        )


@router.get("", response_model=list[TrickCatalogRead])
async def list_tricks(
    session: DbSession,
    current_user: CurrentUser,
    division: Division | None = None,
    search: str | None = Query(default=None, max_length=128),
) -> list[TrickCatalogRead]:
    query = (
        select(TrickCatalogORM)
        .options(selectinload(TrickCatalogORM.examples).selectinload(TrickExampleORM.video))
        .where(TrickCatalogORM.owner_id == current_user.id)
        .order_by(TrickCatalogORM.name)
    )
    if division is not None:
        query = query.where(TrickCatalogORM.division == division)
    if search and search.strip():
        term = f"%{search.strip().lower()}%"
        query = query.where(
            or_(
                func.lower(TrickCatalogORM.name).like(term),
                func.lower(TrickCatalogORM.description).like(term),
            )
        )
    result = await session.execute(query)
    return [_catalog_read(trick) for trick in result.scalars().unique().all()]


@router.post("", response_model=TrickCatalogRead, status_code=status.HTTP_201_CREATED)
async def create_trick(
    payload: TrickCatalogCreate, session: DbSession, current_user: CurrentUser
) -> TrickCatalogRead:
    await _assert_unique_name(session, current_user.id, payload.division, payload.name)
    trick = TrickCatalogORM(owner_id=current_user.id, **payload.model_dump())
    session.add(trick)
    await session.commit()
    return _catalog_read(await _get_owned_trick(session, current_user, trick.id))


@router.patch("/{trick_id}", response_model=TrickCatalogRead)
async def update_trick(
    trick_id: str,
    payload: TrickCatalogUpdate,
    session: DbSession,
    current_user: CurrentUser,
) -> TrickCatalogRead:
    trick = await _get_owned_trick(session, current_user, trick_id)
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    if "name" in changes:
        await _assert_unique_name(
            session, current_user.id, trick.division, str(changes["name"]), exclude_id=trick.id
        )
    if "aliases" in changes:
        changes["aliases"] = list(
            dict.fromkeys(value.strip() for value in changes["aliases"] if value.strip())
        )
    for field_name, value in changes.items():
        setattr(trick, field_name, value)
    await session.commit()
    return _catalog_read(await _get_owned_trick(session, current_user, trick.id))


@router.delete("/{trick_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_trick(trick_id: str, session: DbSession, current_user: CurrentUser) -> Response:
    trick = await _get_owned_trick(session, current_user, trick_id)
    await session.delete(trick)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{trick_id}/examples",
    response_model=TrickExampleRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_trick_example(
    trick_id: str,
    payload: TrickExampleCreate,
    session: DbSession,
    current_user: CurrentUser,
) -> TrickExampleRead:
    trick = await _get_owned_trick(session, current_user, trick_id)
    video_result = await session.execute(
        select(VideoAssetORM).where(
            VideoAssetORM.id == payload.video_id,
            VideoAssetORM.owner_id == current_user.id,
            VideoAssetORM.deleted_at.is_(None),
        )
    )
    video = video_result.scalar_one_or_none()
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")
    if video.division != trick.division:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The example video division must match the trick division.",
        )
    _validate_example_window(video, payload.start_ms, payload.end_ms)
    if payload.is_primary:
        await session.execute(
            update(TrickExampleORM)
            .where(TrickExampleORM.trick_id == trick.id)
            .values(is_primary=False)
        )
    example = TrickExampleORM(
        trick_id=trick.id,
        created_by=current_user.id,
        **payload.model_dump(),
    )
    session.add(example)
    await session.commit()
    await session.refresh(example)
    example.video = video
    return _example_read(example)


async def _get_owned_example(
    session: DbSession,
    current_user: CurrentUser,
    trick_id: str,
    example_id: str,
) -> tuple[TrickCatalogORM, TrickExampleORM]:
    trick = await _get_owned_trick(session, current_user, trick_id)
    example = next((item for item in trick.examples if item.id == example_id), None)
    if example is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Example not found.")
    return trick, example


@router.patch("/{trick_id}/examples/{example_id}", response_model=TrickExampleRead)
async def update_trick_example(
    trick_id: str,
    example_id: str,
    payload: TrickExampleUpdate,
    session: DbSession,
    current_user: CurrentUser,
) -> TrickExampleRead:
    trick, example = await _get_owned_example(session, current_user, trick_id, example_id)
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    start_ms = int(changes.get("start_ms", example.start_ms))
    end_ms = int(changes.get("end_ms", example.end_ms))
    _validate_example_window(example.video, start_ms, end_ms)
    if changes.get("is_primary"):
        await session.execute(
            update(TrickExampleORM)
            .where(TrickExampleORM.trick_id == trick.id)
            .values(is_primary=False)
        )
    for field_name, value in changes.items():
        setattr(example, field_name, value)
    await session.commit()
    await session.refresh(example)
    return _example_read(example)


@router.delete("/{trick_id}/examples/{example_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_trick_example(
    trick_id: str,
    example_id: str,
    session: DbSession,
    current_user: CurrentUser,
) -> Response:
    _, example = await _get_owned_example(session, current_user, trick_id, example_id)
    await session.delete(example)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
