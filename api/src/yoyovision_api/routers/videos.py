"""Video upload, listing, retrieval, download, and deletion."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, UploadFile, status
from fastapi.responses import Response
from sqlalchemy import select
from yoyovision_ml.media_validation import ALLOWED_MIME_TYPES

from yoyovision_api.db_models import AnalysisJobORM, VideoAssetORM
from yoyovision_api.deps import CurrentUser, DbSession, OwnedVideo, SettingsDep, StorageDep
from yoyovision_api.schemas import AnalysisJobCreate, AnalysisJobRead, VideoAssetRead
from yoyovision_api.security import MediaValidationError
from yoyovision_api.services.job_service import create_and_dispatch_analysis_job
from yoyovision_api.services.video_service import create_video_from_upload

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("", response_model=VideoAssetRead, status_code=status.HTTP_201_CREATED)
async def upload_video(
    session: DbSession,
    storage: StorageDep,
    settings: SettingsDep,
    current_user: CurrentUser,
    file: UploadFile,
) -> VideoAssetORM:
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "unsupported_mime_type",
                "message": f"'{file.content_type}' is not an allowed video MIME type.",
            },
        )

    file_bytes = await file.read()
    try:
        video = await create_video_from_upload(
            session=session,
            storage=storage,
            settings=settings,
            owner=current_user,
            original_filename=file.filename or "upload",
            declared_mime_type=file.content_type,
            file_bytes=file_bytes,
        )
    except MediaValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    await create_and_dispatch_analysis_job(session, settings, video)
    await session.commit()
    return video


@router.get("", response_model=list[VideoAssetRead])
async def list_videos(session: DbSession, current_user: CurrentUser) -> list[VideoAssetORM]:
    result = await session.execute(
        select(VideoAssetORM)
        .where(VideoAssetORM.owner_id == current_user.id, VideoAssetORM.deleted_at.is_(None))
        .order_by(VideoAssetORM.created_at.desc())
    )
    return list(result.scalars().all())


@router.get("/download-proxy")
async def download_proxy(
    key: str, session: DbSession, storage: StorageDep, current_user: CurrentUser
) -> Response:
    """Streams a video's bytes through an authenticated endpoint.

    Used by the local-filesystem storage backend, which has no real
    signed-URL mechanism; this endpoint re-validates ownership from the
    storage key itself so a filesystem path is never exposed to, or trusted
    from, the client. Registered before the `/{video_id}` routes below so
    the literal "download-proxy" path segment is never mistakenly captured
    as a `video_id` path parameter.
    """
    result = await session.execute(
        select(VideoAssetORM).where(
            VideoAssetORM.storage_key == key,
            VideoAssetORM.owner_id == current_user.id,
            VideoAssetORM.deleted_at.is_(None),
        )
    )
    video = result.scalar_one_or_none()
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found.")
    data = storage.get(video.storage_key)
    return Response(content=data, media_type=video.mime_type)


@router.get("/{video_id}", response_model=VideoAssetRead)
async def get_video(video: OwnedVideo) -> VideoAssetORM:
    return video


@router.get("/{video_id}/stream")
async def stream_video(video: OwnedVideo, storage: StorageDep) -> Response:
    """Streams a video's bytes for HTML5 `<video>` playback, keyed by
    `video_id` (never by storage key -- `VideoAssetRead` deliberately omits
    `storage_key` per the "do not expose storage paths" security
    requirement). Ownership is already enforced by the `OwnedVideo`
    dependency. In production with the S3-compatible backend, prefer
    redirecting the client to a short-lived signed URL instead of proxying
    bytes through this process; this direct-streaming path is what the
    local-filesystem dev backend needs since it has no signed-URL concept.
    """
    data = storage.get(video.storage_key)
    return Response(content=data, media_type=video.mime_type)


@router.get("/{video_id}/analyses", response_model=list[AnalysisJobRead])
async def list_video_analyses(video: OwnedVideo, session: DbSession) -> list[AnalysisJobORM]:
    result = await session.execute(
        select(AnalysisJobORM)
        .where(AnalysisJobORM.video_id == video.id)
        .order_by(AnalysisJobORM.created_at.desc())
    )
    return list(result.scalars().all())


@router.post(
    "/{video_id}/analyses",
    response_model=AnalysisJobRead,
    status_code=status.HTTP_201_CREATED,
)
async def trigger_video_analysis(
    video: OwnedVideo,
    session: DbSession,
    settings: SettingsDep,
    payload: AnalysisJobCreate | None = None,
    shadow: bool = Query(
        default=False,
        description=(
            "If true, runs this analysis in shadow mode (Prompt F): the full "
            "pipeline still runs and persists real events/deductions/score, "
            "but the job is flagged `is_shadow` so clients know not to treat "
            "it as the video's official/canonical result."
        ),
    ),
) -> AnalysisJobORM:
    """Manually (re-)triggers analysis for a video (e.g. after a failed run,
    or to try a new model version in shadow mode)."""
    job = await create_and_dispatch_analysis_job(
        session,
        settings,
        video,
        is_shadow=shadow,
        pipeline_adapter_config=(
            payload.pipeline_adapter_config if payload is not None else None
        ),
    )
    await session.commit()
    return job


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video: OwnedVideo,
    session: DbSession,
    storage: StorageDep,
    hard: bool = Query(
        default=False,
        description=(
            "If true, permanently removes the video's DB rows (and all derived "
            "analysis data) in addition to its stored bytes. If false (default), "
            "the video bytes and derived artefacts are still deleted, but a "
            "soft-deleted metadata tombstone is kept for audit purposes."
        ),
    ),
) -> Response:
    if storage.exists(video.storage_key):
        storage.delete(video.storage_key)

    if hard:
        await session.delete(video)
    else:
        video.deleted_at = datetime.now(UTC)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
