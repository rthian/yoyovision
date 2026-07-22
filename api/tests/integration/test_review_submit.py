"""Integration tests for analysis submit lock and dataset record export."""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from yoyovision_ml.domain import JobStatus
from yoyovision_ml.media_validation import VideoMetadata

from yoyovision_api import security
from yoyovision_api.db_models import AnalysisJobORM

_MP4_HEADER = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8
_MP4_BODY = _MP4_HEADER + b"\x00" * 1024


@pytest.fixture(autouse=True)
def _mock_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_metadata = VideoMetadata(
        duration_ms=12_000, width=1280, height=720, fps=30.0, video_codec="h264"
    )
    monkeypatch.setattr(security, "probe_video_metadata", lambda path: fake_metadata)


async def _upload_analysis_with_one_event(
    client: AsyncClient, auth_headers: dict[str, str]
) -> tuple[str, str]:
    upload = await client.post(
        "/videos", headers=auth_headers, files={"file": ("a.mp4", _MP4_BODY, "video/mp4")}
    )
    analyses = await client.get(f"/videos/{upload.json()['id']}/analyses", headers=auth_headers)
    analysis_id = str(analyses.json()[0]["id"])
    event = await client.post(
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
    return analysis_id, str(event.json()["id"])


async def _mark_completed(db_session: AsyncSession, analysis_id: str) -> None:
    job = await db_session.get(AnalysisJobORM, analysis_id)
    assert job is not None
    job.status = JobStatus.COMPLETED
    await db_session.commit()


async def test_submit_locks_event_mutations(
    client: AsyncClient,
    auth_headers: dict[str, str],
    db_session: AsyncSession,
) -> None:
    analysis_id, event_id = await _upload_analysis_with_one_event(client, auth_headers)
    await _mark_completed(db_session, analysis_id)

    submit = await client.post(f"/analyses/{analysis_id}/submit", headers=auth_headers)
    assert submit.status_code == 200
    assert submit.json()["review_state"] == "submitted"
    assert submit.json()["submitted_at"] is not None

    patch = await client.patch(
        f"/analyses/{analysis_id}/events/{event_id}",
        headers=auth_headers,
        json={"label": "mount_edited"},
    )
    assert patch.status_code == 409


async def test_reopen_restores_editing(
    client: AsyncClient,
    auth_headers: dict[str, str],
    db_session: AsyncSession,
) -> None:
    analysis_id, event_id = await _upload_analysis_with_one_event(client, auth_headers)
    await _mark_completed(db_session, analysis_id)
    await client.post(f"/analyses/{analysis_id}/submit", headers=auth_headers)

    reopen = await client.post(f"/analyses/{analysis_id}/reopen", headers=auth_headers)
    assert reopen.status_code == 200
    assert reopen.json()["review_state"] == "draft"

    patch = await client.patch(
        f"/analyses/{analysis_id}/events/{event_id}",
        headers=auth_headers,
        json={"label": "mount_edited"},
    )
    assert patch.status_code == 200
    assert patch.json()["label"] == "mount_edited"


async def test_export_dataset_record_matches_schema(
    client: AsyncClient,
    auth_headers: dict[str, str],
    db_session: AsyncSession,
) -> None:
    analysis_id, _ = await _upload_analysis_with_one_event(client, auth_headers)
    await _mark_completed(db_session, analysis_id)

    response = await client.get(
        f"/analyses/{analysis_id}/export/dataset-record.json", headers=auth_headers
    )
    assert response.status_code == 200
    payload = json.loads(response.text)
    assert payload["schema_version"] == "1.0.0"
    assert payload["ontology_version"] == "dataset-ontology-v1"
    assert payload["video"]["duration_ms"] == 12_000
    assert len(payload["trick_events"]) == 1
    assert payload["trick_events"][0]["label"] == "mount_1"
    assert payload["is_adjudicated"] is False

    await client.post(f"/analyses/{analysis_id}/submit", headers=auth_headers)
    submitted = await client.get(
        f"/analyses/{analysis_id}/export/dataset-record.json", headers=auth_headers
    )
    assert submitted.json()["is_adjudicated"] is True
