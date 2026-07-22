"""Integration tests for the manual Freestyle Evaluation entry endpoint (MVP
scope: "Freestyle Evaluation placeholders and manual values")."""

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


async def test_get_evaluation_before_any_entry_returns_null(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_and_get_analysis_id(client, auth_headers)
    response = await client.get(f"/analyses/{analysis_id}/evaluation", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() is None


async def test_upsert_evaluation_persists_partial_manual_entry_and_updates_score(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_and_get_analysis_id(client, auth_headers)

    response = await client.put(
        f"/analyses/{analysis_id}/evaluation",
        headers=auth_headers,
        json={"execution": 8.0, "control": 7.5, "notes": "Strong opening, shaky landing."},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["execution"] == 8.0
    assert body["control"] == 7.5
    assert body["trick_diversity"] is None
    assert body["source"] == "human"

    score = await client.get(f"/analyses/{analysis_id}/score", headers=auth_headers)
    score_body = score.json()
    assert score_body["freestyle_evaluation_raw"] == 15.5
    assert any("missing manual values" in w for w in score_body["warnings"])


async def test_upsert_evaluation_twice_updates_the_same_row(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    analysis_id = await _upload_and_get_analysis_id(client, auth_headers)

    await client.put(
        f"/analyses/{analysis_id}/evaluation", headers=auth_headers, json={"execution": 5.0}
    )
    second = await client.put(
        f"/analyses/{analysis_id}/evaluation", headers=auth_headers, json={"execution": 9.0}
    )
    assert second.json()["execution"] == 9.0

    fetched = await client.get(f"/analyses/{analysis_id}/evaluation", headers=auth_headers)
    assert fetched.json()["execution"] == 9.0
