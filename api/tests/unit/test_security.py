"""Unit tests for `yoyovision_api.security`: storage key generation and the
upload validation chain (with `ffprobe` mocked out, since it isn't
installed in this sandbox -- see `docs/architecture.md`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from yoyovision_ml.media_validation import MediaValidationError, VideoMetadata

from yoyovision_api import security

_MP4_HEADER = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8


def test_generate_storage_key_is_server_derived_and_safe() -> None:
    key = security.generate_storage_key("owner-123", "video/mp4")
    assert key.startswith("videos/owner-123/")
    assert key.endswith(".mp4")
    assert ".." not in key


def test_generate_storage_key_ignores_unknown_mime_extension() -> None:
    key = security.generate_storage_key("owner-123", "application/octet-stream")
    assert key.endswith(".bin")


def test_validate_uploaded_video_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_metadata = VideoMetadata(
        duration_ms=15_000, width=1920, height=1080, fps=30.0, video_codec="h264"
    )
    monkeypatch.setattr(security, "probe_video_metadata", lambda path: fake_metadata)

    result = security.validate_uploaded_video(
        local_path=tmp_path / "upload.mp4",
        head_bytes=_MP4_HEADER,
        declared_mime_type="video/mp4",
        file_size_bytes=1_000_000,
        max_upload_bytes=500_000_000,
        max_duration_ms=600_000,
    )
    assert result.metadata.duration_ms == 15_000


def test_validate_uploaded_video_rejects_bad_signature(tmp_path: Path) -> None:
    with pytest.raises(MediaValidationError) as exc_info:
        security.validate_uploaded_video(
            local_path=tmp_path / "upload.mp4",
            head_bytes=b"not a real video",
            declared_mime_type="video/mp4",
            file_size_bytes=1_000,
            max_upload_bytes=500_000_000,
            max_duration_ms=600_000,
        )
    assert exc_info.value.code == "signature_unrecognized"


def test_validate_uploaded_video_rejects_oversized_file(tmp_path: Path) -> None:
    with pytest.raises(MediaValidationError) as exc_info:
        security.validate_uploaded_video(
            local_path=tmp_path / "upload.mp4",
            head_bytes=_MP4_HEADER,
            declared_mime_type="video/mp4",
            file_size_bytes=1_000_000_000,
            max_upload_bytes=500_000_000,
            max_duration_ms=600_000,
        )
    assert exc_info.value.code == "file_too_large"


def test_validate_uploaded_video_rejects_too_long_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_metadata = VideoMetadata(
        duration_ms=900_000, width=1920, height=1080, fps=30.0, video_codec="h264"
    )
    monkeypatch.setattr(security, "probe_video_metadata", lambda path: fake_metadata)

    with pytest.raises(MediaValidationError) as exc_info:
        security.validate_uploaded_video(
            local_path=tmp_path / "upload.mp4",
            head_bytes=_MP4_HEADER,
            declared_mime_type="video/mp4",
            file_size_bytes=1_000_000,
            max_upload_bytes=500_000_000,
            max_duration_ms=600_000,
        )
    assert exc_info.value.code == "duration_too_long"
