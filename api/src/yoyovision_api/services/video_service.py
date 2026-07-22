"""Video upload orchestration: validation, server-side storage key
generation, and persistence of the `VideoAssetORM` row.

Security requirements enforced here (see spec "SECURITY REQUIREMENTS"):
MIME/signature validation, size/duration limits, server-generated storage
keys (never derived from the client filename), and never trusting the
client-declared content type alone.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from yoyovision_ml.domain import VideoStatus
from yoyovision_ml.interfaces import StoragePort

from yoyovision_api.config import Settings
from yoyovision_api.db_models import User, VideoAssetORM
from yoyovision_api.security import (
    MediaValidationError,
    generate_storage_key,
    validate_uploaded_video,
)

__all__ = ["MediaValidationError", "create_video_from_upload"]

_HEAD_SNIFF_BYTES = 64


async def create_video_from_upload(
    session: AsyncSession,
    storage: StoragePort,
    settings: Settings,
    owner: User,
    original_filename: str,
    declared_mime_type: str,
    file_bytes: bytes,
) -> VideoAssetORM:
    """Validates and persists an uploaded video. Raises `MediaValidationError`
    (mapped to HTTP 422 by the router) if any security/quality check fails.

    The file is written to a temporary path only to run `ffprobe` against it
    (ffprobe requires a real file path); the temporary file is always removed,
    and it is never executed.
    """
    with tempfile.NamedTemporaryFile(suffix=".upload", delete=True) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(file_bytes)
        tmp.flush()

        validation = validate_uploaded_video(
            local_path=tmp_path,
            head_bytes=file_bytes[:_HEAD_SNIFF_BYTES],
            declared_mime_type=declared_mime_type,
            file_size_bytes=len(file_bytes),
            max_upload_bytes=settings.storage_max_upload_bytes,
            max_duration_ms=settings.storage_max_duration_ms,
        )

    storage_key = generate_storage_key(owner.id, declared_mime_type)
    storage.put(storage_key, file_bytes, declared_mime_type)

    video = VideoAssetORM(
        owner_id=owner.id,
        original_filename=_truncate_filename(original_filename),
        storage_key=storage_key,
        mime_type=declared_mime_type,
        duration_ms=validation.metadata.duration_ms,
        width=validation.metadata.width,
        height=validation.metadata.height,
        fps=validation.metadata.fps,
        file_size=len(file_bytes),
        status=VideoStatus.READY,
    )
    session.add(video)
    await session.flush()
    return video


def _truncate_filename(original_filename: str) -> str:
    """Stores the client filename as inert display metadata only (never used
    to build a filesystem/storage path); still bounded in length."""
    safe = original_filename.replace("\x00", "")
    return safe[:512]
