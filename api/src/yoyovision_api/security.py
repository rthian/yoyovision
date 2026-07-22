"""Server-side security helpers for uploads: storage key generation and
orchestration of the `yoyovision_ml.media_validation` checks.

Storage keys are ALWAYS generated server-side from a fresh UUID plus the
owning user's id -- the client-provided filename is never used to build a
path (it is only stored as free-text metadata after sanitization checks).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from yoyovision_ml.media_validation import (
    MediaValidationError,
    VideoMetadata,
    probe_video_metadata,
    validate_duration,
    validate_signature,
    validate_size,
)

__all__ = [
    "MediaValidationError",
    "UploadValidationResult",
    "generate_storage_key",
    "validate_uploaded_video",
]

_EXTENSION_BY_MIME = {
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "video/webm": "webm",
}


def generate_storage_key(owner_id: str, mime_type: str) -> str:
    """Server-generated, collision-resistant, path-traversal-safe key.
    Never derived from a client-supplied filename."""
    extension = _EXTENSION_BY_MIME.get(mime_type, "bin")
    return f"videos/{owner_id}/{uuid.uuid4()}.{extension}"


@dataclass(slots=True, frozen=True)
class UploadValidationResult:
    metadata: VideoMetadata


def validate_uploaded_video(
    local_path: Path,
    head_bytes: bytes,
    declared_mime_type: str,
    file_size_bytes: int,
    max_upload_bytes: int,
    max_duration_ms: int,
) -> UploadValidationResult:
    """Runs the full validation chain; raises `MediaValidationError` on any
    failure with a machine-readable `code` suitable for API error responses."""
    validate_signature(head_bytes, declared_mime_type)
    validate_size(file_size_bytes, max_upload_bytes)
    metadata = probe_video_metadata(local_path)
    validate_duration(metadata.duration_ms, max_duration_ms)
    return UploadValidationResult(metadata=metadata)
