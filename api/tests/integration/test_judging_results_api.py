"""Integration tests for judging entry results aggregation."""

from __future__ import annotations

from urllib.parse import urlparse

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from yoyovision_ml.domain import JobStatus, Source, VideoStatus
from yoyovision_ml.media_validation import VideoMetadata

from yoyovision_api import security
from yoyovision_api.db_models import (
    AnalysisJobORM,
    FreestyleEvaluationORM,
    JudgeFreestyleScoreORM,
    User,
    VideoAssetORM,
)

_MP4_HEADER = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8
_MP4_BODY = _MP4_HEADER + b"\x00" * 1024

_FE = {
    "execution": 8.0,
    "control": 7.0,
    "trick_diversity": 8.0,
    "space_use_emphasis": 7.0,
    "music_choreography": 8.0,
    "music_construction": 7.0,
    "body_control": 8.0,
    "showmanship": 9.0,
    "notes": "done",
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


def _token_from_url(invite_url: str) -> str:
    return invite_url.rstrip("/").rsplit("/", 1)[-1]


async def _create_open_entry(
    client: AsyncClient, admin_headers: dict[str, str]
) -> tuple[str, str]:
    video_id = await _upload_video(client, admin_headers)
    entry = (
        await client.post(
            "/judging-entries",
            headers=admin_headers,
            json={"title": "Results test", "mode": "training", "video_ids": [video_id]},
        )
    ).json()
    entry_id = entry["id"]
    entry_video_id = entry["videos"][0]["id"]
    await client.patch(
        f"/judging-entries/{entry_id}",
        headers=admin_headers,
        json={"status": "open"},
    )
    return entry_id, entry_video_id


async def _add_judge_token(
    client: AsyncClient, admin_headers: dict[str, str], entry_id: str, name: str
) -> str:
    invite = (
        await client.post(
            f"/judging-entries/{entry_id}/judges",
            headers=admin_headers,
            json={"display_name": name},
        )
    ).json()
    return _token_from_url(invite["invite_url"])


async def _submit_fe(client: AsyncClient, token: str, entry_video_id: str, **overrides) -> None:
    payload = {**_FE, **overrides}
    response = await client.post(
        f"/judge-access/{token}/videos/{entry_video_id}/submit",
        json=payload,
    )
    assert response.status_code == 200, response.text


async def _link_analysis_with_fe(
    db_session: AsyncSession,
    admin_user: User,
    video_id: str,
    *,
    execution: float = 7.0,
    control: float = 7.0,
    is_shadow: bool = False,
) -> str:
    job = AnalysisJobORM(
        video_id=video_id,
        status=JobStatus.COMPLETED,
        progress=1.0,
        pipeline_version="0.1.0-dev",
        is_shadow=is_shadow,
    )
    db_session.add(job)
    await db_session.flush()
    db_session.add(
        FreestyleEvaluationORM(
            analysis_id=job.id,
            execution=execution,
            control=control,
            trick_diversity=7.0,
            space_use_emphasis=7.0,
            music_choreography=7.0,
            music_construction=7.0,
            body_control=7.0,
            showmanship=7.0,
            source=Source.MODEL,
            notes="ai",
        )
    )
    await db_session.commit()
    return job.id


async def test_non_admin_cannot_fetch_results(
    client: AsyncClient, auth_headers: dict[str, str], admin_headers: dict[str, str]
) -> None:
    entry_id, _ = await _create_open_entry(client, admin_headers)
    response = await client.get(f"/judging-entries/{entry_id}/results", headers=auth_headers)
    assert response.status_code == 403


async def test_simple_mean_panel_aggregate(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    entry_id, entry_video_id = await _create_open_entry(client, admin_headers)
    token_a = await _add_judge_token(client, admin_headers, entry_id, "Alex")
    token_b = await _add_judge_token(client, admin_headers, entry_id, "Blake")
    await _submit_fe(client, token_a, entry_video_id, execution=6.0)
    await _submit_fe(client, token_b, entry_video_id, execution=8.0)

    results = await client.get(f"/judging-entries/{entry_id}/results", headers=admin_headers)
    assert results.status_code == 200, results.text
    video = results.json()["videos"][0]
    assert video["panel_aggregate"]["execution"] == 7.0
    assert len(video["judges"]) == 2


async def test_shadow_judge_excluded_from_aggregate(
    client: AsyncClient, admin_headers: dict[str, str]
) -> None:
    entry_id, entry_video_id = await _create_open_entry(client, admin_headers)
    token_a = await _add_judge_token(client, admin_headers, entry_id, "Alex")
    shadow_invite = (
        await client.post(
            f"/judging-entries/{entry_id}/judges",
            headers=admin_headers,
            json={"display_name": "Shadow", "is_shadow": True},
        )
    ).json()
    token_shadow = _token_from_url(shadow_invite["invite_url"])
    await _submit_fe(client, token_a, entry_video_id, execution=6.0)
    await _submit_fe(client, token_shadow, entry_video_id, execution=10.0)

    video = (
        await client.get(f"/judging-entries/{entry_id}/results", headers=admin_headers)
    ).json()["videos"][0]
    assert video["panel_aggregate"]["execution"] == 6.0
    shadow_row = next(row for row in video["judges"] if row["display_name"] == "Shadow")
    assert shadow_row["included_in_aggregate"] is False


async def test_profile_b_gap_fills_from_ai(
    client: AsyncClient,
    admin_headers: dict[str, str],
    db_session: AsyncSession,
    admin_user: User,
) -> None:
    entry_id, entry_video_id = await _create_open_entry(client, admin_headers)
    detail = await client.get(f"/judging-entries/{entry_id}", headers=admin_headers)
    video_id = detail.json()["videos"][0]["video_id"]
    analysis_id = await _link_analysis_with_fe(db_session, admin_user, video_id, control=7.5)
    await client.patch(
        f"/judging-entries/{entry_id}/videos/{entry_video_id}/analyses",
        headers=admin_headers,
        json={"official_analysis_id": analysis_id},
    )
    await client.patch(
        f"/judging-entries/{entry_id}",
        headers=admin_headers,
        json={"ai_mix_profile": "B"},
    )

    invite = (
        await client.post(
            f"/judging-entries/{entry_id}/judges",
            headers=admin_headers,
            json={"display_name": "Alex"},
        )
    ).json()
    assignment_id = invite["assignment_id"]
    db_session.add(
        JudgeFreestyleScoreORM(
            assignment_id=assignment_id,
            entry_video_id=entry_video_id,
            execution=8.0,
            control=None,
            trick_diversity=8.0,
            space_use_emphasis=7.0,
            music_choreography=8.0,
            music_construction=7.0,
            body_control=8.0,
            showmanship=9.0,
            notes="partial human",
            is_submitted=True,
        )
    )
    await db_session.commit()

    video = (
        await client.get(f"/judging-entries/{entry_id}/results", headers=admin_headers)
    ).json()["videos"][0]
    assert video["panel_aggregate"]["control"] == 7.5
    assert "control" in video["ai_filled_categories"]


async def test_profile_c_includes_ai_virtual_judge(
    client: AsyncClient,
    admin_headers: dict[str, str],
    db_session: AsyncSession,
    admin_user: User,
) -> None:
    entry_id, entry_video_id = await _create_open_entry(client, admin_headers)
    video_id = (
        await client.get(f"/judging-entries/{entry_id}", headers=admin_headers)
    ).json()["videos"][0]["video_id"]
    analysis_id = await _link_analysis_with_fe(db_session, admin_user, video_id, execution=9.0)
    await client.patch(
        f"/judging-entries/{entry_id}/videos/{entry_video_id}/analyses",
        headers=admin_headers,
        json={"official_analysis_id": analysis_id},
    )
    await client.patch(
        f"/judging-entries/{entry_id}",
        headers=admin_headers,
        json={"ai_mix_profile": "C", "aggregation_mode": "simple_mean"},
    )

    token_a = await _add_judge_token(client, admin_headers, entry_id, "Alex")
    token_b = await _add_judge_token(client, admin_headers, entry_id, "Blake")
    await _submit_fe(client, token_a, entry_video_id, execution=6.0)
    await _submit_fe(client, token_b, entry_video_id, execution=8.0)

    video = (
        await client.get(f"/judging-entries/{entry_id}/results", headers=admin_headers)
    ).json()["videos"][0]
    assert video["ai_virtual_judge_included"] is True
    assert video["panel_aggregate"]["execution"] == pytest.approx(7.666666, rel=1e-3)
