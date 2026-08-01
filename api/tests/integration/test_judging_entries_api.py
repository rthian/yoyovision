"""Integration tests for admin multi-judge entry APIs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from yoyovision_ml.media_validation import VideoMetadata

from yoyovision_api import security
from yoyovision_api.auth import hash_password
from yoyovision_api.db_models import JudgeAssignmentORM, User
from yoyovision_api.judging_enums import UserRole
from yoyovision_api.services.invite_token import hash_token, is_token_active

_MP4_HEADER = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8
_MP4_BODY = _MP4_HEADER + b"\x00" * 1024


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


async def test_non_admin_cannot_create_judging_entry(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    video_id = await _upload_video(client, auth_headers)
    response = await client.post(
        "/judging-entries",
        headers=auth_headers,
        json={"title": "Round 1", "mode": "training", "video_ids": [video_id]},
    )
    assert response.status_code == 403


async def test_admin_can_create_entry_add_judge_and_rotate_invite(
    client: AsyncClient,
    admin_headers: dict[str, str],
    db_session: AsyncSession,
) -> None:
    video_id = await _upload_video(client, admin_headers)
    create = await client.post(
        "/judging-entries",
        headers=admin_headers,
        json={
            "title": "Prelims",
            "mode": "contest",
            "video_ids": [video_id],
            "ai_mix_profile": "A",
            "aggregation_mode": "auto",
        },
    )
    assert create.status_code == 201, create.text
    entry = create.json()
    assert entry["title"] == "Prelims"
    assert len(entry["videos"]) == 1

    invite = await client.post(
        f"/judging-entries/{entry['id']}/judges",
        headers=admin_headers,
        json={"display_name": "Alex"},
    )
    assert invite.status_code == 201, invite.text
    body = invite.json()
    assert body["display_name"] == "Alex"
    assert "/judge/" in body["invite_url"]
    assert "Alex" in body["share_message"]

    rotate = await client.post(
        f"/judging-entries/{entry['id']}/judges/{body['assignment_id']}/rotate",
        headers=admin_headers,
    )
    assert rotate.status_code == 200, rotate.text
    rotated = rotate.json()
    assert rotated["invite_url"] != body["invite_url"]

    detail = await client.get(f"/judging-entries/{entry['id']}", headers=admin_headers)
    assert detail.status_code == 200
    assert len(detail.json()["judges"]) == 1


async def test_revoked_invite_token_is_inactive(
    client: AsyncClient,
    admin_headers: dict[str, str],
    db_session: AsyncSession,
) -> None:
    video_id = await _upload_video(client, admin_headers)
    entry_id = (
        await client.post(
            "/judging-entries",
            headers=admin_headers,
            json={"title": "Revoke test", "mode": "training", "video_ids": [video_id]},
        )
    ).json()["id"]
    invite = (
        await client.post(
            f"/judging-entries/{entry_id}/judges",
            headers=admin_headers,
            json={"display_name": "Sam"},
        )
    ).json()

    revoke = await client.post(
        f"/judging-entries/{entry_id}/judges/{invite['assignment_id']}/revoke",
        headers=admin_headers,
    )
    assert revoke.status_code == 204

    assignment = await db_session.get(JudgeAssignmentORM, invite["assignment_id"])
    assert assignment is not None
    assert assignment.revoked_at is not None
    assert not is_token_active(
        token_expires_at=assignment.token_expires_at,
        revoked_at=assignment.revoked_at,
    )


async def test_expired_token_is_inactive(db_session: AsyncSession) -> None:
    assignment = JudgeAssignmentORM(
        entry_id="missing",
        display_name="Expired",
        invite_token_hash=hash_token("expired-token"),
        token_prefix="expired-",
        token_expires_at=datetime.now(UTC) - timedelta(hours=1),
        include_in_results=True,
        is_shadow=False,
    )
    assert not is_token_active(
        token_expires_at=assignment.token_expires_at,
        revoked_at=None,
        now=datetime.now(UTC),
    )
