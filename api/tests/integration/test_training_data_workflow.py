"""Integration coverage for YouTube provenance and human trick labels."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from yoyovision_ml.media_validation import VideoMetadata

from yoyovision_api import security
from yoyovision_api.services import youtube_import_service

_MP4_BODY = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1024


@pytest.fixture(autouse=True)
def _mock_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = VideoMetadata(
        duration_ms=12_000, width=1280, height=720, fps=30.0, video_codec="h264"
    )
    monkeypatch.setattr(security, "probe_video_metadata", lambda path: metadata)


@pytest.fixture
def mock_youtube_download(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _download(url: str, *, max_bytes: int, max_duration_ms: int):
        del url, max_bytes, max_duration_ms
        return youtube_import_service.DownloadedYoutubeVideo(
            data=_MP4_BODY,
            title="World Yo-Yo Contest.mp4",
            mime_type="video/mp4",
            canonical_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            external_id="dQw4w9WgXcQ",
        )

    monkeypatch.setattr(youtube_import_service, "download_youtube_video", _download)


async def _import_video(client: AsyncClient, headers: dict[str, str]) -> dict[str, object]:
    response = await client.post(
        "/videos/youtube",
        headers=headers,
        json={
            "url": "https://youtu.be/dQw4w9WgXcQ",
            "division": "2A",
            "player_id": "competitor-042",
            "rights_confirmed": True,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


async def test_youtube_import_persists_provenance_without_running_1a_model(
    client: AsyncClient,
    auth_headers: dict[str, str],
    mock_youtube_download: None,
) -> None:
    video = await _import_video(client, auth_headers)
    assert video["division"] == "2A"
    assert video["source_type"] == "youtube"
    assert video["source_external_id"] == "dQw4w9WgXcQ"
    assert video["rights_confirmed_at"] is not None

    analyses = await client.get(f"/videos/{video['id']}/analyses", headers=auth_headers)
    assert analyses.json() == []


async def test_youtube_import_requires_rights_confirmation(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/videos/youtube",
        headers=auth_headers,
        json={
            "url": "https://youtu.be/dQw4w9WgXcQ",
            "division": "1A",
            "player_id": "competitor-042",
            "rights_confirmed": False,
        },
    )
    assert response.status_code == 422


async def test_duplicate_youtube_video_is_rejected(
    client: AsyncClient,
    auth_headers: dict[str, str],
    mock_youtube_download: None,
) -> None:
    await _import_video(client, auth_headers)
    duplicate = await client.post(
        "/videos/youtube",
        headers=auth_headers,
        json={
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "division": "2A",
            "player_id": "competitor-042",
            "rights_confirmed": True,
        },
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "youtube_video_already_imported"


async def test_annotations_crud_and_training_record_export(
    client: AsyncClient,
    auth_headers: dict[str, str],
    mock_youtube_download: None,
) -> None:
    video = await _import_video(client, auth_headers)
    video_id = str(video["id"])
    create = await client.post(
        f"/videos/{video_id}/annotations",
        headers=auth_headers,
        json={
            "label": "Two-hand alternating loops",
            "element_type": "loop_combo",
            "start_ms": 1200,
            "end_ms": 2800,
            "outcome": "success",
            "technical_credit": "positive_click",
            "notes": "Clean synchronized finish",
        },
    )
    assert create.status_code == 201, create.text
    annotation = create.json()
    assert annotation["division"] == "2A"

    update = await client.patch(
        f"/videos/{video_id}/annotations/{annotation['id']}",
        headers=auth_headers,
        json={"end_ms": 3000, "technical_credit": "no_click"},
    )
    assert update.status_code == 200, update.text
    assert update.json()["end_ms"] == 3000

    export = await client.get(f"/videos/{video_id}/training-record", headers=auth_headers)
    assert export.status_code == 200, export.text
    record = export.json()
    assert record["video"]["division"] == "2A"
    assert record["video"]["player_id"] == "competitor-042"
    assert record["video"]["rights_confirmed"] is True
    assert record["trick_events"][0]["element_type"] == "loop_combo"
    assert record["trick_events"][0]["technical_credit"] == "no_click"

    delete = await client.delete(
        f"/videos/{video_id}/annotations/{annotation['id']}", headers=auth_headers
    )
    assert delete.status_code == 204


def test_youtube_url_parser_rejects_non_youtube_hosts() -> None:
    with pytest.raises(youtube_import_service.YoutubeImportError):
        youtube_import_service.parse_youtube_video_id(
            "https://example.com/watch?v=dQw4w9WgXcQ"
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
    ],
)
def test_youtube_url_parser_accepts_single_video_forms(url: str) -> None:
    assert youtube_import_service.parse_youtube_video_id(url) == "dQw4w9WgXcQ"
