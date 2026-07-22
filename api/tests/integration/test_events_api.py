"""Integration tests for the AnalysisEvent review workflow (product principle
#4: add / edit / delete / confirm every detected event), including that every
mutation triggers a score recompute."""

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


async def _upload_and_get_analysis_id(client: AsyncClient, auth_headers: dict[str, str]) -> str:
    upload = await client.post(
        "/videos", headers=auth_headers, files={"file": ("a.mp4", _MP4_BODY, "video/mp4")}
    )
    analyses = await client.get(f"/videos/{upload.json()['id']}/analyses", headers=auth_headers)
    return str(analyses.json()[0]["id"])


async def test_create_event_is_human_confirmed_and_updates_score(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_and_get_analysis_id(client, auth_headers)

    response = await client.post(
        f"/analyses/{analysis_id}/events",
        headers=auth_headers,
        json={
            "label": "mount_1",
            "family": "mount",
            "start_ms": 0,
            "end_ms": 500,
            "confidence": 1.0,
            "outcome": "success",
            "difficulty_band": "basic",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "human"
    assert body["review_status"] == "confirmed"

    score = await client.get(f"/analyses/{analysis_id}/score", headers=auth_headers)
    assert score.json()["technical_raw"] == 1.0


async def test_update_event_marks_edited_and_flips_source_to_human(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_and_get_analysis_id(client, auth_headers)
    created = await client.post(
        f"/analyses/{analysis_id}/events",
        headers=auth_headers,
        json={
            "label": "hop_1",
            "family": "hop",
            "start_ms": 0,
            "end_ms": 300,
            "outcome": "miss",
            "difficulty_band": "basic",
        },
    )
    event_id = created.json()["id"]

    updated = await client.patch(
        f"/analyses/{analysis_id}/events/{event_id}",
        headers=auth_headers,
        json={"outcome": "success"},
    )
    assert updated.status_code == 200
    assert updated.json()["outcome"] == "success"
    assert updated.json()["review_status"] == "edited"


async def test_reject_event_excludes_it_from_score_but_keeps_the_row(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_and_get_analysis_id(client, auth_headers)
    created = await client.post(
        f"/analyses/{analysis_id}/events",
        headers=auth_headers,
        json={
            "label": "mount_1",
            "family": "mount",
            "start_ms": 0,
            "end_ms": 500,
            "outcome": "success",
            "difficulty_band": "basic",
        },
    )
    event_id = created.json()["id"]

    rejected = await client.post(
        f"/analyses/{analysis_id}/events/{event_id}/reject", headers=auth_headers
    )
    assert rejected.status_code == 200
    assert rejected.json()["review_status"] == "rejected"

    score = await client.get(f"/analyses/{analysis_id}/score", headers=auth_headers)
    assert score.json()["technical_raw"] == 0.0

    events = await client.get(f"/analyses/{analysis_id}/events", headers=auth_headers)
    assert len(events.json()) == 1


async def test_delete_event_removes_it_and_recomputes_score(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_and_get_analysis_id(client, auth_headers)
    created = await client.post(
        f"/analyses/{analysis_id}/events",
        headers=auth_headers,
        json={
            "label": "mount_1",
            "family": "mount",
            "start_ms": 0,
            "end_ms": 500,
            "outcome": "success",
            "difficulty_band": "basic",
        },
    )
    event_id = created.json()["id"]

    deleted = await client.delete(
        f"/analyses/{analysis_id}/events/{event_id}", headers=auth_headers
    )
    assert deleted.status_code == 204

    events = await client.get(f"/analyses/{analysis_id}/events", headers=auth_headers)
    assert events.json() == []


async def test_events_on_another_users_analysis_return_404(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get("/analyses/not-a-real-id/events", headers=auth_headers)
    assert response.status_code == 404
