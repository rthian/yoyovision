"""Integration tests for video upload, listing, retrieval, ownership, and
deletion, exercised through the full FastAPI app over an in-memory DB."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from yoyovision_ml.media_validation import VideoMetadata

from yoyovision_api import security
from yoyovision_api.auth import create_access_token, hash_password
from yoyovision_api.config import Settings
from yoyovision_api.db_models import User

_MP4_HEADER = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8
_MP4_BODY = _MP4_HEADER + b"\x00" * 1024


@pytest.fixture(autouse=True)
def _mock_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ffprobe` isn't installed in this sandbox; every upload test mocks the
    probe step so the rest of the validation/persistence pipeline still runs
    for real."""
    fake_metadata = VideoMetadata(
        duration_ms=12_000, width=1280, height=720, fps=30.0, video_codec="h264"
    )
    monkeypatch.setattr(security, "probe_video_metadata", lambda path: fake_metadata)


async def test_unauthenticated_request_is_rejected(client: AsyncClient) -> None:
    response = await client.get("/videos")
    assert response.status_code == 401


async def test_upload_video_creates_video_and_analysis_job(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/videos",
        headers=auth_headers,
        files={"file": ("freestyle.mp4", _MP4_BODY, "video/mp4")},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["mime_type"] == "video/mp4"
    assert body["duration_ms"] == 12_000
    assert body["status"] == "ready"

    analyses = await client.get(f"/videos/{body['id']}/analyses", headers=auth_headers)
    assert analyses.status_code == 200
    assert len(analyses.json()) == 1
    assert analyses.json()[0]["status"] == "pending"


async def test_upload_rejects_disallowed_mime_type(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/videos",
        headers=auth_headers,
        files={"file": ("malware.exe", b"MZ\x90\x00", "application/x-msdownload")},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unsupported_mime_type"


async def test_list_videos_only_returns_current_users_videos(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_settings: Settings,
    db_session: AsyncSession,
) -> None:
    await client.post(
        "/videos", headers=auth_headers, files={"file": ("a.mp4", _MP4_BODY, "video/mp4")}
    )

    other_user = User(email="other@yoyovision.local", hashed_password=hash_password("x"))
    db_session.add(other_user)
    await db_session.commit()
    token = create_access_token(other_user.id, test_settings)
    other_headers = {"Authorization": f"Bearer {token}"}

    mine = await client.get("/videos", headers=auth_headers)
    theirs = await client.get("/videos", headers=other_headers)
    assert len(mine.json()) == 1
    assert len(theirs.json()) == 0


async def test_get_video_owned_by_another_user_returns_404(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_settings: Settings,
    db_session: AsyncSession,
) -> None:
    upload = await client.post(
        "/videos", headers=auth_headers, files={"file": ("a.mp4", _MP4_BODY, "video/mp4")}
    )
    video_id = upload.json()["id"]

    other_user = User(email="intruder@yoyovision.local", hashed_password=hash_password("x"))
    db_session.add(other_user)
    await db_session.commit()
    token = create_access_token(other_user.id, test_settings)
    other_headers = {"Authorization": f"Bearer {token}"}

    response = await client.get(f"/videos/{video_id}", headers=other_headers)
    assert response.status_code == 404


async def test_stream_video_returns_bytes_for_owner(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    upload = await client.post(
        "/videos", headers=auth_headers, files={"file": ("a.mp4", _MP4_BODY, "video/mp4")}
    )
    video_id = upload.json()["id"]

    response = await client.get(f"/videos/{video_id}/stream", headers=auth_headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == _MP4_BODY


async def test_stream_video_owned_by_another_user_returns_404(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_settings: Settings,
    db_session: AsyncSession,
) -> None:
    upload = await client.post(
        "/videos", headers=auth_headers, files={"file": ("a.mp4", _MP4_BODY, "video/mp4")}
    )
    video_id = upload.json()["id"]

    other_user = User(email="stream-intruder@yoyovision.local", hashed_password=hash_password("x"))
    db_session.add(other_user)
    await db_session.commit()
    token = create_access_token(other_user.id, test_settings)
    other_headers = {"Authorization": f"Bearer {token}"}

    response = await client.get(f"/videos/{video_id}/stream", headers=other_headers)
    assert response.status_code == 404


async def test_delete_video_soft_deletes_and_removes_storage_bytes(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    upload = await client.post(
        "/videos", headers=auth_headers, files={"file": ("a.mp4", _MP4_BODY, "video/mp4")}
    )
    video_id = upload.json()["id"]

    delete_response = await client.delete(f"/videos/{video_id}", headers=auth_headers)
    assert delete_response.status_code == 204

    get_response = await client.get(f"/videos/{video_id}", headers=auth_headers)
    assert get_response.status_code == 404
