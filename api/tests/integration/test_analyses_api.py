"""Integration tests for analysis-job retrieval and cancellation (Prompt F),
exercised through the full FastAPI app over an in-memory DB."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from yoyovision_ml.media_validation import VideoMetadata

from yoyovision_api import security
from yoyovision_api.auth import create_access_token, hash_password
from yoyovision_api.config import Settings
from yoyovision_api.db_models import User

_MP4_HEADER = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8
_MP4_BODY = _MP4_HEADER + b"\x00" * 1024


@pytest.fixture(autouse=True)
def _mock_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_metadata = VideoMetadata(
        duration_ms=12_000, width=1280, height=720, fps=30.0, video_codec="h264"
    )
    monkeypatch.setattr(security, "probe_video_metadata", lambda path: fake_metadata)


async def _upload_and_get_job_id(client: AsyncClient, auth_headers: dict[str, str]) -> str:
    upload = await client.post(
        "/videos", headers=auth_headers, files={"file": ("a.mp4", _MP4_BODY, "video/mp4")}
    )
    video_id = upload.json()["id"]
    analyses = await client.get(f"/videos/{video_id}/analyses", headers=auth_headers)
    return str(analyses.json()[0]["id"])


async def test_cancel_analysis_sets_cancel_requested(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    job_id = await _upload_and_get_job_id(client, auth_headers)

    response = await client.post(f"/analyses/{job_id}/cancel", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["cancel_requested"] is True

    refetched = await client.get(f"/analyses/{job_id}", headers=auth_headers)
    assert refetched.json()["cancel_requested"] is True


async def test_delete_analysis_removes_finished_job(
    client: AsyncClient, auth_headers: dict[str, str], db_session: object
) -> None:
    from yoyovision_ml.domain import JobStatus

    from yoyovision_api.db_models import AnalysisJobORM

    job_id = await _upload_and_get_job_id(client, auth_headers)
    job = await db_session.get(AnalysisJobORM, job_id)  # type: ignore[attr-defined]
    assert job is not None
    job.status = JobStatus.COMPLETED
    await db_session.commit()  # type: ignore[attr-defined]

    response = await client.delete(f"/analyses/{job_id}", headers=auth_headers)
    assert response.status_code == 204

    missing = await client.get(f"/analyses/{job_id}", headers=auth_headers)
    assert missing.status_code == 404


async def test_delete_running_analysis_returns_409(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    job_id = await _upload_and_get_job_id(client, auth_headers)

    response = await client.delete(f"/analyses/{job_id}", headers=auth_headers)
    assert response.status_code == 409


async def test_cancel_analysis_owned_by_another_user_returns_404(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_settings: Settings,
    db_session: object,
) -> None:
    job_id = await _upload_and_get_job_id(client, auth_headers)

    other_user = User(email="cancel-intruder@yoyovision.local", hashed_password=hash_password("x"))
    db_session.add(other_user)  # type: ignore[attr-defined]
    await db_session.commit()  # type: ignore[attr-defined]
    token = create_access_token(other_user.id, test_settings)
    other_headers = {"Authorization": f"Bearer {token}"}

    response = await client.post(f"/analyses/{job_id}/cancel", headers=other_headers)
    assert response.status_code == 404


async def test_trigger_analysis_with_shadow_true_flags_job_as_shadow(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    upload = await client.post(
        "/videos", headers=auth_headers, files={"file": ("a.mp4", _MP4_BODY, "video/mp4")}
    )
    video_id = upload.json()["id"]

    response = await client.post(
        f"/videos/{video_id}/analyses", params={"shadow": True}, headers=auth_headers
    )
    assert response.status_code == 201
    assert response.json()["is_shadow"] is True

    default_response = await client.post(
        f"/videos/{video_id}/analyses", headers=auth_headers
    )
    assert default_response.json()["is_shadow"] is False


async def test_patch_analysis_ruleset_updates_version_and_recomputes(
    client: AsyncClient, auth_headers: dict[str, str], db_session: object
) -> None:
    from yoyovision_ml.domain import JobStatus

    from yoyovision_api.db_models import AnalysisJobORM, ScoreBreakdownORM

    job_id = await _upload_and_get_job_id(client, auth_headers)
    job = await db_session.get(AnalysisJobORM, job_id)  # type: ignore[attr-defined]
    assert job is not None
    job.status = JobStatus.COMPLETED
    await db_session.commit()  # type: ignore[attr-defined]

    response = await client.patch(
        f"/analyses/{job_id}/ruleset",
        headers=auth_headers,
        json={"ruleset_version": "iyyf-wyyc-25-draft"},
    )
    assert response.status_code == 200
    assert response.json()["ruleset_version"] == "iyyf-wyyc-25-draft"

    score = await client.get(f"/analyses/{job_id}/score", headers=auth_headers)
    assert score.status_code == 200
    assert score.json()["ruleset_version"] == "iyyf-wyyc-25-draft"

