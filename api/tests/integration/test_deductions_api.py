"""Integration tests for the MajorDeduction review workflow (yo-yo stop /
change / detach and other equipment-related deductions), mirroring the event
review workflow's audit-trail guarantees."""

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


async def test_create_deduction_is_human_confirmed_and_applies_points(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_and_get_analysis_id(client, auth_headers)

    response = await client.post(
        f"/analyses/{analysis_id}/deductions",
        headers=auth_headers,
        json={"type": "yoyo_stop", "timestamp_ms": 4_000, "quantity": 1, "points": 2.0},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "human"
    assert body["review_status"] == "confirmed"

    score = await client.get(f"/analyses/{analysis_id}/score", headers=auth_headers)
    assert score.json()["major_deductions"] == 2.0


async def test_update_deduction_marks_edited(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_and_get_analysis_id(client, auth_headers)
    created = await client.post(
        f"/analyses/{analysis_id}/deductions",
        headers=auth_headers,
        json={"type": "yoyo_stop", "timestamp_ms": 1_000, "quantity": 1, "points": 2.0},
    )
    deduction_id = created.json()["id"]

    updated = await client.patch(
        f"/analyses/{analysis_id}/deductions/{deduction_id}",
        headers=auth_headers,
        json={"quantity": 2},
    )
    assert updated.status_code == 200
    assert updated.json()["quantity"] == 2
    assert updated.json()["review_status"] == "edited"


async def test_reject_deduction_excludes_it_from_score(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_and_get_analysis_id(client, auth_headers)
    created = await client.post(
        f"/analyses/{analysis_id}/deductions",
        headers=auth_headers,
        json={"type": "yoyo_detach", "timestamp_ms": 2_000, "quantity": 1, "points": 4.0},
    )
    deduction_id = created.json()["id"]

    rejected = await client.post(
        f"/analyses/{analysis_id}/deductions/{deduction_id}/reject", headers=auth_headers
    )
    assert rejected.status_code == 200
    assert rejected.json()["review_status"] == "rejected"

    score = await client.get(f"/analyses/{analysis_id}/score", headers=auth_headers)
    assert score.json()["major_deductions"] == 0.0


async def test_confirm_deduction_does_not_change_points(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_and_get_analysis_id(client, auth_headers)
    created = await client.post(
        f"/analyses/{analysis_id}/deductions",
        headers=auth_headers,
        json={"type": "yoyo_change", "timestamp_ms": 3_000, "quantity": 1, "points": 3.0},
    )
    deduction_id = created.json()["id"]

    confirmed = await client.post(
        f"/analyses/{analysis_id}/deductions/{deduction_id}/confirm", headers=auth_headers
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["review_status"] == "confirmed"
    assert confirmed.json()["points"] == 3.0


async def test_delete_deduction_removes_it_and_recomputes_score(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_and_get_analysis_id(client, auth_headers)
    created = await client.post(
        f"/analyses/{analysis_id}/deductions",
        headers=auth_headers,
        json={"type": "yoyo_stop", "timestamp_ms": 5_000, "quantity": 1, "points": 2.0},
    )
    deduction_id = created.json()["id"]

    deleted = await client.delete(
        f"/analyses/{analysis_id}/deductions/{deduction_id}", headers=auth_headers
    )
    assert deleted.status_code == 204

    deductions = await client.get(f"/analyses/{analysis_id}/deductions", headers=auth_headers)
    assert deductions.json() == []
