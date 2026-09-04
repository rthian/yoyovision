"""Integration tests for judge timestamp clicks (Phase F)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from yoyovision_ml.media_validation import VideoMetadata

from yoyovision_api import security

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


def _token_from_invite_url(invite_url: str) -> str:
    return invite_url.rstrip("/").rsplit("/", 1)[-1]


async def _training_entry_with_clicker(
    client: AsyncClient, admin_headers: dict[str, str]
) -> tuple[str, str, str]:
    video_id = await _upload_video(client, admin_headers)
    entry = (
        await client.post(
            "/judging-entries",
            headers=admin_headers,
            json={
                "title": "Clicker test",
                "mode": "training",
                "video_ids": [video_id],
                "click_mode": "training_only",
            },
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
            json={"display_name": "Sam"},
        )
    ).json()
    return entry_id, entry_video_id, _token_from_invite_url(invite["invite_url"])


async def test_judge_can_add_and_delete_clicks(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    _, entry_video_id, token = await _training_entry_with_clicker(client, admin_headers)

    access = await client.get(f"/judge-access/{token}")
    assert access.status_code == 200
    assert access.json()["click_mode"] == "training_only"

    created = await client.post(
        f"/judge-access/{token}/videos/{entry_video_id}/clicks",
        json={"timestamp_ms": 1500, "label": "rock the baby"},
    )
    assert created.status_code == 201, created.text
    click_id = created.json()["id"]

    listed = await client.get(f"/judge-access/{token}/videos/{entry_video_id}/clicks")
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["timestamp_ms"] == 1500

    deleted = await client.delete(f"/judge-access/{token}/clicks/{click_id}")
    assert deleted.status_code == 204

    listed_after = await client.get(f"/judge-access/{token}/videos/{entry_video_id}/clicks")
    assert listed_after.json() == []


async def test_clicker_disabled_when_mode_off(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    video_id = await _upload_video(client, admin_headers)
    entry = (
        await client.post(
            "/judging-entries",
            headers=admin_headers,
            json={"title": "No clicker", "mode": "training", "video_ids": [video_id]},
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
            json={"display_name": "Sam"},
        )
    ).json()
    token = _token_from_invite_url(invite["invite_url"])

    response = await client.post(
        f"/judge-access/{token}/videos/{entry_video_id}/clicks",
        json={"timestamp_ms": 1000},
    )
    assert response.status_code == 403


async def test_admin_calibration_endpoint(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    entry_id, entry_video_id, token = await _training_entry_with_clicker(
        client, admin_headers
    )
    await client.post(
        f"/judge-access/{token}/videos/{entry_video_id}/clicks",
        json={"timestamp_ms": 2000},
    )

    calibration = await client.get(
        f"/judging-entries/{entry_id}/calibration", headers=admin_headers
    )
    assert calibration.status_code == 200, calibration.text
    body = calibration.json()
    assert body["click_mode"] == "training_only"
    assert body["videos"][0]["panel_click_count"] == 1
