"""Validated YouTube ingestion for user-authorized training footage."""

from __future__ import annotations

import asyncio
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from yoyovision_ml.media_validation import sniff_container_mime_type

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
_YOUTUBE_FORMAT_SELECTOR = "/".join(
    (
        # Prefer a progressive file when YouTube exposes one. Many current
        # uploads expose only separate DASH video/audio streams, though, and
        # YoYoVision's visual analysis does not require an audio track. The
        # video-only fallbacks avoid requiring ffmpeg just to merge audio.
        "best[ext=mp4][vcodec!=none][acodec!=none][height<=720]",
        "best[ext=webm][vcodec!=none][acodec!=none][height<=720]",
        "bestvideo[ext=mp4][vcodec^=avc1][height<=720]",
        "bestvideo[ext=mp4][height<=720]",
        "bestvideo[ext=webm][height<=720]",
        "bestvideo[height<=720]",
    )
)


class YoutubeImportError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True, frozen=True)
class DownloadedYoutubeVideo:
    data: bytes
    title: str
    mime_type: str
    canonical_url: str
    external_id: str


def parse_youtube_video_id(url: str) -> str:
    """Returns an 11-character video ID without allowing arbitrary hosts."""
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in _YOUTUBE_HOSTS:
        raise YoutubeImportError(
            "unsupported_video_url", "Enter an HTTPS youtube.com or youtu.be video URL."
        )

    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    elif parsed.path.startswith("/shorts/") or parsed.path.startswith("/embed/"):
        video_id = parsed.path.split("/")[2]
    else:
        video_id = ""

    if not _VIDEO_ID_RE.fullmatch(video_id):
        raise YoutubeImportError("invalid_youtube_video_id", "The YouTube video ID is invalid.")
    return video_id


async def download_youtube_video(
    url: str, *, max_bytes: int, max_duration_ms: int
) -> DownloadedYoutubeVideo:
    video_id = parse_youtube_video_id(url)
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    return await asyncio.to_thread(
        _download_sync,
        canonical_url,
        video_id,
        max_bytes,
        max_duration_ms,
    )


def _download_sync(
    canonical_url: str, video_id: str, max_bytes: int, max_duration_ms: int
) -> DownloadedYoutubeVideo:
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError as exc:  # pragma: no cover - deployment configuration failure
        raise YoutubeImportError(
            "youtube_import_unavailable", "YouTube importing is not installed on this server."
        ) from exc

    try:
        with tempfile.TemporaryDirectory(prefix="yoyovision-youtube-") as directory:
            output_template = str(Path(directory) / "video.%(ext)s")
            options = {
                "format": _YOUTUBE_FORMAT_SELECTOR,
                "outtmpl": output_template,
                "noplaylist": True,
                "quiet": True,
                "noprogress": True,
                "no_warnings": True,
                # Recent yt-dlp versions need a JavaScript runtime for the
                # complete set of YouTube formats. Node is already required by
                # the frontend and is therefore available in normal installs.
                "js_runtimes": {"node": {}},
                "max_filesize": max_bytes,
                "restrictfilenames": True,
                "socket_timeout": 30,
                "retries": 2,
                "fragment_retries": 2,
            }
            with YoutubeDL(options) as downloader:
                metadata = downloader.extract_info(canonical_url, download=False)
                if metadata is None:
                    raise YoutubeImportError("youtube_metadata_missing", "No video metadata found.")
                duration_ms = int(float(metadata.get("duration") or 0) * 1000)
                if duration_ms <= 0 or duration_ms > max_duration_ms:
                    raise YoutubeImportError(
                        "duration_not_allowed",
                        f"Video duration must be between 1ms and {max_duration_ms}ms.",
                    )
                declared_size = metadata.get("filesize") or metadata.get("filesize_approx")
                if declared_size and int(declared_size) > max_bytes:
                    raise YoutubeImportError(
                        "video_too_large", f"Video exceeds the {max_bytes}-byte import limit."
                    )
                info = downloader.extract_info(canonical_url, download=True)
                downloaded_path = Path(downloader.prepare_filename(info))

            if not downloaded_path.exists():
                candidates = list(Path(directory).glob("video.*"))
                if len(candidates) != 1:
                    raise YoutubeImportError(
                        "youtube_download_missing", "The downloaded video file was not found."
                    )
                downloaded_path = candidates[0]
            data = downloaded_path.read_bytes()
            if len(data) > max_bytes:
                raise YoutubeImportError(
                    "video_too_large", f"Video exceeds the {max_bytes}-byte import limit."
                )
            mime_type = sniff_container_mime_type(data[:64])
            if mime_type not in {"video/mp4", "video/webm"}:
                raise YoutubeImportError(
                    "unsupported_download_format", "YouTube returned an unsupported container."
                )
            title = str(info.get("title") or f"youtube-{video_id}")[:480]
            return DownloadedYoutubeVideo(
                data=data,
                title=f"{title}.{downloaded_path.suffix.lstrip('.')}",
                mime_type=mime_type,
                canonical_url=canonical_url,
                external_id=video_id,
            )
    except YoutubeImportError:
        raise
    except DownloadError as exc:
        raise YoutubeImportError(
            "youtube_download_failed", "YouTube could not provide this video."
        ) from exc
