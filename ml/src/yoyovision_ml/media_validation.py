"""Media validation: MIME/signature sniffing, ffprobe metadata, quality checks.

Security requirements enforced here: validate MIME type AND file signature
(magic bytes) rather than trusting the client-declared content type or the
filename extension; enforce file-size and duration limits; never execute an
uploaded file (ffprobe is invoked read-only, with a timeout, against a path
already written to a server-controlled storage location).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: (declared/allowed mime type) -> tuple of acceptable magic-byte signatures.
#: MP4/MOV both use the ISO base media container ("ftyp" box after 4 size bytes);
#: WebM/Matroska uses an EBML header.
_MP4_MOV_FTYP_OFFSET = 4
_MP4_MOV_FTYP_MAGIC = b"ftyp"
_WEBM_EBML_MAGIC = b"\x1a\x45\xdf\xa3"

ALLOWED_MIME_TYPES: frozenset[str] = frozenset({"video/mp4", "video/quicktime", "video/webm"})

_FFPROBE_TIMEOUT_SECONDS = 20


class MediaValidationError(Exception):
    """Raised when an uploaded file fails signature, size, or duration checks."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def sniff_container_mime_type(head_bytes: bytes) -> str | None:
    """Best-effort container sniffing from the first bytes of a file.

    Returns one of `ALLOWED_MIME_TYPES` or `None` if unrecognized. This is a
    lightweight signature check, not a full demux -- it protects against a
    mislabeled/renamed file, not against a maliciously crafted container that
    also happens to carry a valid ftyp/EBML header.
    """
    if (
        len(head_bytes) >= _MP4_MOV_FTYP_OFFSET + 4
        and head_bytes[_MP4_MOV_FTYP_OFFSET : _MP4_MOV_FTYP_OFFSET + 4] == _MP4_MOV_FTYP_MAGIC
    ):
        # Disambiguate mp4 vs mov by the ftyp major-brand field when present.
        brand = head_bytes[8:12] if len(head_bytes) >= 12 else b""
        if brand in (b"qt  ",):
            return "video/quicktime"
        return "video/mp4"
    if head_bytes[:4] == _WEBM_EBML_MAGIC:
        return "video/webm"
    return None


def validate_signature(head_bytes: bytes, declared_mime_type: str) -> None:
    """Raises `MediaValidationError` if the declared MIME type doesn't match
    the sniffed container signature, or if neither is an allowed type."""
    if declared_mime_type not in ALLOWED_MIME_TYPES:
        raise MediaValidationError(
            "unsupported_mime_type",
            f"'{declared_mime_type}' is not an allowed video MIME type.",
        )
    sniffed = sniff_container_mime_type(head_bytes)
    if sniffed is None:
        raise MediaValidationError(
            "signature_unrecognized",
            "File signature does not match any supported video container.",
        )
    if sniffed != declared_mime_type:
        raise MediaValidationError(
            "signature_mismatch",
            f"Declared MIME type '{declared_mime_type}' does not match detected "
            f"container signature '{sniffed}'.",
        )


def validate_size(file_size_bytes: int, max_bytes: int) -> None:
    if file_size_bytes <= 0:
        raise MediaValidationError("empty_file", "Uploaded file is empty.")
    if file_size_bytes > max_bytes:
        raise MediaValidationError(
            "file_too_large",
            f"File size {file_size_bytes} bytes exceeds limit of {max_bytes} bytes.",
        )


@dataclass(slots=True, frozen=True)
class VideoMetadata:
    duration_ms: int
    width: int
    height: int
    fps: float
    video_codec: str


def probe_video_metadata(path: Path) -> VideoMetadata:
    """Reads container/stream metadata via `ffprobe`. Read-only; never executes
    the uploaded file as code. Requires `ffmpeg`/`ffprobe` on PATH (provided in
    the worker/API Docker images)."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, read-only inspection
            command,
            capture_output=True,
            timeout=_FFPROBE_TIMEOUT_SECONDS,
            check=True,
        )
    except FileNotFoundError as exc:
        raise MediaValidationError(
            "ffprobe_not_found", "ffprobe is not installed or not on PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaValidationError("ffprobe_timeout", "ffprobe timed out inspecting file.") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaValidationError(
            "ffprobe_failed", f"ffprobe could not parse file: {exc.stderr!r}"
        ) from exc

    payload = json.loads(result.stdout)
    video_streams = [s for s in payload.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        raise MediaValidationError("no_video_stream", "No video stream found in file.")
    stream = video_streams[0]

    duration_s_raw = payload.get("format", {}).get("duration") or stream.get("duration")
    if duration_s_raw is None:
        raise MediaValidationError("duration_unknown", "Could not determine video duration.")
    duration_ms = int(round(float(duration_s_raw) * 1000))

    fps = 0.0
    frame_rate_raw = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if frame_rate_raw and frame_rate_raw != "0/0":
        numerator, _, denominator = frame_rate_raw.partition("/")
        denom_value = float(denominator) if denominator else 1.0
        if denom_value:
            fps = float(numerator) / denom_value

    return VideoMetadata(
        duration_ms=duration_ms,
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        fps=round(fps, 3),
        video_codec=str(stream.get("codec_name", "unknown")),
    )


def validate_duration(duration_ms: int, max_duration_ms: int) -> None:
    if duration_ms <= 0:
        raise MediaValidationError("zero_duration", "Video duration could not be determined.")
    if duration_ms > max_duration_ms:
        raise MediaValidationError(
            "duration_too_long",
            f"Video duration {duration_ms}ms exceeds limit of {max_duration_ms}ms.",
        )
