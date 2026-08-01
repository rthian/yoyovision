"""FastAPI dependency injection: DB sessions, current user, storage backend."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import for side-effect: registers "local"/"s3" storage backends with the ml registry.
from yoyovision_ml import storage as _storage_adapters  # noqa: F401
from yoyovision_ml.adapters_registry import create_storage_backend
from yoyovision_ml.interfaces import StoragePort

from yoyovision_api.auth import AuthError, decode_access_token
from yoyovision_api.config import Settings, get_settings
from yoyovision_api.db import get_db_session
from yoyovision_api.db_models import AnalysisJobORM, JudgeAssignmentORM, User, VideoAssetORM
from yoyovision_api.judging_enums import UserRole

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

SettingsDep = Annotated[Settings, Depends(get_settings)]
DbSession = Annotated[AsyncSession, Depends(get_db_session)]


async def get_current_user(
    token: Annotated[str, Depends(_oauth2_scheme)],
    session: DbSession,
    settings: SettingsDep,
) -> User:
    try:
        user_id = decode_access_token(token, settings)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_storage(settings: SettingsDep) -> StoragePort:
    if settings.storage_backend == "local":
        backend = create_storage_backend("local", root=settings.storage_local_root)
    elif settings.storage_backend == "s3":
        backend = create_storage_backend(
            "s3",
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            use_path_style=settings.s3_use_path_style,
        )
    else:
        raise ValueError(f"Unknown STORAGE_BACKEND: {settings.storage_backend!r}")
    return backend  # type: ignore[return-value]


StorageDep = Annotated[StoragePort, Depends(get_storage)]


async def get_owned_video(
    video_id: str, session: DbSession, current_user: CurrentUser
) -> VideoAssetORM:
    """Fetches a video and enforces ownership. Returns 404 (not 403) for
    videos owned by another user, so existence of another user's video is
    never revealed."""
    result = await session.execute(select(VideoAssetORM).where(VideoAssetORM.id == video_id))
    video = result.scalar_one_or_none()
    if video is None or video.owner_id != current_user.id or video.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")
    return video


async def get_owned_job(
    analysis_id: str, session: DbSession, current_user: CurrentUser
) -> AnalysisJobORM:
    """Fetches an analysis job and enforces ownership through its parent
    video. Returns 404 (not 403) if the job doesn't exist or the parent
    video isn't owned by the current user."""
    result = await session.execute(
        select(AnalysisJobORM)
        .join(VideoAssetORM, AnalysisJobORM.video_id == VideoAssetORM.id)
        .where(AnalysisJobORM.id == analysis_id, VideoAssetORM.owner_id == current_user.id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    return job


OwnedVideo = Annotated[VideoAssetORM, Depends(get_owned_video)]
OwnedJob = Annotated[AnalysisJobORM, Depends(get_owned_job)]


async def get_current_admin(current_user: CurrentUser) -> User:
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")
    return current_user


CurrentAdmin = Annotated[User, Depends(get_current_admin)]
