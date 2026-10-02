"""Integration coverage for the cross-view trick catalog."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from yoyovision_ml.media_validation import VideoMetadata

from yoyovision_api import security

_MP4_BODY = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1024


@pytest.fixture(autouse=True)
def _mock_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = VideoMetadata(
        duration_ms=30_000, width=1280, height=720, fps=30.0, video_codec="h264"
    )
    monkeypatch.setattr(security, "probe_video_metadata", lambda path: metadata)


async def _upload_video(
    client: AsyncClient, headers: dict[str, str], *, division: str = "1A"
) -> dict[str, object]:
    response = await client.post(
        "/videos",
        headers=headers,
        files={"file": ("tutorial.mp4", _MP4_BODY, "video/mp4")},
        data={
            "division": division,
            "player_id": "mir-kim",
            "rights_confirmed": "true",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


async def test_trick_catalog_groups_cross_view_examples(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    video = await _upload_video(client, auth_headers)
    create = await client.post(
        "/tricks",
        headers=auth_headers,
        json={
            "name": "Mir Kim Over the Head",
            "division": "1A",
            "aliases": ["Over the Head", "Over the Head"],
            "description": "Tutorial and stage references.",
        },
    )
    assert create.status_code == 201, create.text
    trick = create.json()
    assert trick["aliases"] == ["Over the Head"]

    tutorial = await client.post(
        f"/tricks/{trick['id']}/examples",
        headers=auth_headers,
        json={
            "video_id": video["id"],
            "start_ms": 1_000,
            "end_ms": 4_500,
            "view_type": "tutorial",
            "camera_angle": "front",
            "playback_speed": 1.0,
            "is_primary": True,
        },
    )
    assert tutorial.status_code == 201, tutorial.text
    assert tutorial.json()["original_filename"] == "tutorial.mp4"

    stage = await client.post(
        f"/tricks/{trick['id']}/examples",
        headers=auth_headers,
        json={
            "video_id": video["id"],
            "start_ms": 10_000,
            "end_ms": 13_000,
            "view_type": "stage",
            "camera_angle": "audience wide",
            "playback_speed": 1.0,
            "is_primary": True,
        },
    )
    assert stage.status_code == 201, stage.text

    listed = await client.get("/tricks?division=1A", headers=auth_headers)
    assert listed.status_code == 200
    payload = listed.json()
    assert len(payload) == 1
    assert len(payload[0]["examples"]) == 2
    assert [example["is_primary"] for example in payload[0]["examples"]] == [False, True]

    remove = await client.delete(
        f"/tricks/{trick['id']}/examples/{tutorial.json()['id']}", headers=auth_headers
    )
    assert remove.status_code == 204
    delete = await client.delete(f"/tricks/{trick['id']}", headers=auth_headers)
    assert delete.status_code == 204
    source_video = await client.get(f"/videos/{video['id']}", headers=auth_headers)
    assert source_video.status_code == 200


async def test_trick_example_validates_division_and_duration(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    video = await _upload_video(client, auth_headers, division="2A")
    trick = (
        await client.post(
            "/tricks",
            headers=auth_headers,
            json={"name": "Over the Head", "division": "1A", "aliases": []},
        )
    ).json()
    mismatch = await client.post(
        f"/tricks/{trick['id']}/examples",
        headers=auth_headers,
        json={
            "video_id": video["id"],
            "start_ms": 0,
            "end_ms": 1_000,
            "view_type": "tutorial",
        },
    )
    assert mismatch.status_code == 422

    one_a_video = await _upload_video(client, auth_headers)
    too_long = await client.post(
        f"/tricks/{trick['id']}/examples",
        headers=auth_headers,
        json={
            "video_id": one_a_video["id"],
            "start_ms": 29_000,
            "end_ms": 31_000,
            "view_type": "stage",
        },
    )
    assert too_long.status_code == 422
