"""Integration tests for JSON/CSV export endpoints: correct payload shape,
the unofficial-score disclaimer, sanitized filenames, and download headers."""

from __future__ import annotations

import csv
import io
import json

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


async def _upload_analysis_with_one_event(client: AsyncClient, auth_headers: dict[str, str]) -> str:
    upload = await client.post(
        "/videos", headers=auth_headers, files={"file": ("a.mp4", _MP4_BODY, "video/mp4")}
    )
    analyses = await client.get(f"/videos/{upload.json()['id']}/analyses", headers=auth_headers)
    analysis_id = str(analyses.json()[0]["id"])
    await client.post(
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
    return analysis_id


async def test_export_json_contains_disclaimer_video_events_and_score(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_analysis_with_one_event(client, auth_headers)

    response = await client.get(f"/analyses/{analysis_id}/export/report.json", headers=auth_headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert "attachment" in response.headers["content-disposition"]

    payload = json.loads(response.text)
    assert "not certified by IYYF" in payload["disclaimer"]
    assert payload["video"]["mime_type"] == "video/mp4"
    assert len(payload["events"]) == 1
    assert payload["events"][0]["label"] == "mount_1"
    assert payload["score"]["technical_raw"] == 1.0


async def test_export_events_csv_has_expected_columns_and_row(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_analysis_with_one_event(client, auth_headers)

    response = await client.get(f"/analyses/{analysis_id}/export/events.csv", headers=auth_headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")

    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 1
    assert rows[0]["label"] == "mount_1"
    assert rows[0]["family"] == "mount"
    assert rows[0]["outcome"] == "success"


async def test_export_deductions_csv_is_empty_but_well_formed_when_no_deductions(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_analysis_with_one_event(client, auth_headers)

    response = await client.get(
        f"/analyses/{analysis_id}/export/deductions.csv", headers=auth_headers
    )
    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert rows == []


async def test_export_filename_is_sanitized_and_server_derived(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_analysis_with_one_event(client, auth_headers)

    response = await client.get(f"/analyses/{analysis_id}/export/report.json", headers=auth_headers)
    disposition = response.headers["content-disposition"]
    assert f"yoyovision-analysis-{analysis_id}.json" in disposition
    assert "/" not in disposition.split("filename=")[-1]


async def test_export_on_nonexistent_analysis_returns_404(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get("/analyses/not-a-real-id/export/report.json", headers=auth_headers)
    assert response.status_code == 404
