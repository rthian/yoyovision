"""Integration tests for token-authenticated judge access APIs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from yoyovision_ml.media_validation import VideoMetadata

from yoyovision_api import security
from yoyovision_api.db_models import JudgeAssignmentORM

_MP4_HEADER = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8
_MP4_BODY = _MP4_HEADER + b"\x00" * 1024

_FE_PAYLOAD = {
    "execution": 8.0,
    "control": 7.5,
    "trick_diversity": 8.5,
    "space_use_emphasis": 7.0,
    "music_choreography": 8.0,
    "music_construction": 7.5,
    "body_control": 8.0,
    "showmanship": 9.0,
    "notes": "solid routine",
}


@pytest.fixture(autouse=True)
def _mock_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_metadata = VideoMetadata(
        duration_ms=12_000, width=1280, height=720, fps=30.0, video_codec="h264"
    )
    monkeypatch.setattr(security, "probe_video_metadata", lambda path: fake_metadata)


async def _upload_video(client: AsyncClient, headers: dict[str, str]) -> str:
    response = await client.post(
        "/videos",
        headers=headers,
        files={"file": ("clip.mp4", _MP4_BODY, "video/mp4")},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _token_from_invite_url(invite_url: str) -> str:
    return invite_url.rstrip("/").rsplit("/", 1)[-1]


async def _open_entry_with_judge(
    client: AsyncClient, admin_headers: dict[str, str]
) -> tuple[str, str, str]:
    video_id = await _upload_video(client, admin_headers)
    entry = (
        await client.post(
            "/judging-entries",
            headers=admin_headers,
            json={"title": "Judge access test", "mode": "training", "video_ids": [video_id]},
        )
    ).json()
    entry_id = entry["id"]
    entry_video_id = entry["videos"][0]["id"]
    await client.patch(
        f"/judging-entries/{entry_id}",
        headers=admin_headers,
        json={"status": "open"},
    )
    invite = (
        await client.post(
            f"/judging-entries/{entry_id}/judges",
            headers=admin_headers,
            json={"display_name": "Alex"},
        )
    ).json()
    return entry_id, entry_video_id, _token_from_invite_url(invite["invite_url"])


async def test_draft_entry_blocks_judge_access(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    video_id = await _upload_video(client, admin_headers)
    entry_id = (
        await client.post(
            "/judging-entries",
            headers=admin_headers,
            json={"title": "Draft", "mode": "training", "video_ids": [video_id]},
        )
    ).json()["id"]
    invite = (
        await client.post(
            f"/judging-entries/{entry_id}/judges",
            headers=admin_headers,
            json={"display_name": "Pat"},
        )
    ).json()
    token = _token_from_invite_url(invite["invite_url"])
    response = await client.get(f"/judge-access/{token}")
    assert response.status_code == 403


async def test_judge_access_happy_path_draft_submit_and_lock(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    _, entry_video_id, token = await _open_entry_with_judge(client, admin_headers)

    access = await client.get(f"/judge-access/{token}")
    assert access.status_code == 200, access.text
    body = access.json()
    assert body["display_name"] == "Alex"
    assert len(body["videos"]) == 1
    assert body["videos"][0]["my_score"] is None
    assert "video_id" not in body["videos"][0]

    draft = await client.put(
        f"/judge-access/{token}/videos/{entry_video_id}/fe",
        json={"execution": 8.0, "notes": "draft"},
    )
    assert draft.status_code == 200, draft.text
    assert draft.json()["is_submitted"] is False

    submit = await client.post(
        f"/judge-access/{token}/videos/{entry_video_id}/submit",
        json=_FE_PAYLOAD,
    )
    assert submit.status_code == 200, submit.text
    assert submit.json()["is_submitted"] is True

    locked = await client.put(
        f"/judge-access/{token}/videos/{entry_video_id}/fe",
        json={"execution": 5.0},
    )
    assert locked.status_code == 409

    reread = await client.get(f"/judge-access/{token}")
    assert reread.json()["videos"][0]["my_score"]["execution"] == 8.0


async def test_judge_stream_returns_bytes(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    _, entry_video_id, token = await _open_entry_with_judge(client, admin_headers)
    response = await client.get(f"/judge-access/{token}/videos/{entry_video_id}/stream")
    assert response.status_code == 200
    assert response.content.startswith(_MP4_HEADER)


async def test_wrong_entry_video_returns_404(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    _, _, token = await _open_entry_with_judge(client, admin_headers)
    response = await client.get(f"/judge-access/{token}/videos/not-a-real-id/stream")
    assert response.status_code == 404


async def test_invalid_token_returns_401(client: AsyncClient) -> None:
    response = await client.get("/judge-access/totally-invalid-token")
    assert response.status_code == 401


async def test_expired_token_returns_410(
    client: AsyncClient,
    admin_headers: dict[str, str],
    db_session: AsyncSession,
) -> None:
    entry_id, _, token = await _open_entry_with_judge(client, admin_headers)
    detail = await client.get(f"/judging-entries/{entry_id}", headers=admin_headers)
    assignment_id = detail.json()["judges"][0]["id"]
    assignment = await db_session.get(JudgeAssignmentORM, assignment_id)
    assert assignment is not None
    assignment.token_expires_at = datetime.now(UTC) - timedelta(hours=1)
    await db_session.commit()

    response = await client.get(f"/judge-access/{token}")
    assert response.status_code == 410


async def test_judge_tokens_are_isolated(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    entry_id, entry_video_id, token_a = await _open_entry_with_judge(client, admin_headers)
    invite_b = (
        await client.post(
            f"/judging-entries/{entry_id}/judges",
            headers=admin_headers,
            json={"display_name": "Blake"},
        )
    ).json()
    token_b = _token_from_invite_url(invite_b["invite_url"])

    await client.post(
        f"/judge-access/{token_a}/videos/{entry_video_id}/submit",
        json=_FE_PAYLOAD,
    )

    access_b = await client.get(f"/judge-access/{token_b}")
    assert access_b.status_code == 200
    assert access_b.json()["display_name"] == "Blake"
    assert access_b.json()["videos"][0]["my_score"] is None


async def test_locked_entry_allows_read_blocks_write(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    entry_id, entry_video_id, token = await _open_entry_with_judge(client, admin_headers)
    await client.patch(
        f"/judging-entries/{entry_id}",
        headers=admin_headers,
        json={"status": "locked"},
    )
    assert (await client.get(f"/judge-access/{token}")).status_code == 200
    write = await client.put(
        f"/judge-access/{token}/videos/{entry_video_id}/fe",
        json={"execution": 5.0},
    )
    assert write.status_code == 403
