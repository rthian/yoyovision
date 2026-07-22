"""Integration tests for `yoyovision_workers.pipeline_runner.run_pipeline_for_job`
against an in-memory SQLite engine and local-filesystem storage, covering the
success path, the missing-job/video guards, and pipeline-failure handling."""

from __future__ import annotations

from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncEngine
from yoyovision_ml.storage import LocalFilesystemStorage

from yoyovision_workers.config import Settings
from yoyovision_workers.pipeline_runner import run_pipeline_for_job
from yoyovision_workers.schema import (
    analysis_events,
    analysis_jobs,
    major_deductions,
    score_breakdowns,
)


async def _fetch_job(engine: AsyncEngine, job_id: str) -> dict[str, Any]:
    async with engine.begin() as conn:
        row = (
            (await conn.execute(select(analysis_jobs).where(analysis_jobs.c.id == job_id)))
            .mappings()
            .first()
        )
        assert row is not None
        return dict(row)


async def test_run_pipeline_for_job_completes_and_persists_events_and_score(
    engine: AsyncEngine,
    settings: Settings,
    storage: LocalFilesystemStorage,
    seeded_job: dict[str, str],
) -> None:
    await run_pipeline_for_job(
        job_id=seeded_job["job_id"], settings=settings, engine=engine, storage=storage
    )

    job = await _fetch_job(engine, seeded_job["job_id"])
    assert job["status"] == "completed"
    assert job["progress"] == 1.0
    assert job["current_stage"] == "done"
    assert job["started_at"] is not None
    assert job["completed_at"] is not None

    async with engine.begin() as conn:
        events = (
            (
                await conn.execute(
                    select(analysis_events).where(
                        analysis_events.c.analysis_id == seeded_job["job_id"]
                    )
                )
            )
            .mappings()
            .all()
        )
        score_row = (
            (
                await conn.execute(
                    select(score_breakdowns).where(
                        score_breakdowns.c.analysis_id == seeded_job["job_id"]
                    )
                )
            )
            .mappings()
            .first()
        )

    assert len(events) > 0
    assert all(e["source"] == "model" for e in events)
    assert all(e["review_status"] == "pending" for e in events)
    assert score_row is not None
    assert score_row["ruleset_version"] == "1a-draft-0.1"
    assert any("not certified by IYYF" in w for w in score_row["warnings"])


async def test_run_pipeline_for_job_persists_deduction_points_from_ruleset(
    engine: AsyncEngine,
    settings: Settings,
    storage: LocalFilesystemStorage,
    seeded_job: dict[str, str],
) -> None:
    await run_pipeline_for_job(
        job_id=seeded_job["job_id"], settings=settings, engine=engine, storage=storage
    )

    async with engine.begin() as conn:
        deductions = (
            (
                await conn.execute(
                    select(major_deductions).where(
                        major_deductions.c.analysis_id == seeded_job["job_id"]
                    )
                )
            )
            .mappings()
            .all()
        )

    for deduction in deductions:
        assert deduction["points"] > 0.0
        assert deduction["source"] == "model"


async def test_run_pipeline_for_job_on_missing_job_is_a_noop(
    engine: AsyncEngine, settings: Settings, storage: LocalFilesystemStorage
) -> None:
    # Should return quietly (logged, not raised) rather than crash the worker.
    await run_pipeline_for_job(
        job_id="does-not-exist", settings=settings, engine=engine, storage=storage
    )


async def test_run_pipeline_for_job_marks_failed_when_video_row_is_missing(
    engine: AsyncEngine, settings: Settings, storage: LocalFilesystemStorage
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            insert(analysis_jobs).values(
                id="orphan-job",
                video_id="missing-video",
                status="pending",
                progress=0.0,
                current_stage="queued",
                pipeline_version="0.1.0-dev",
            )
        )

    await run_pipeline_for_job(
        job_id="orphan-job", settings=settings, engine=engine, storage=storage
    )

    job = await _fetch_job(engine, "orphan-job")
    assert job["status"] == "failed"
    assert job["error_code"] == "video_not_found"


async def test_run_pipeline_for_job_marks_failed_when_storage_get_raises(
    engine: AsyncEngine, settings: Settings, seeded_job: dict[str, str]
) -> None:
    class _BrokenStorage:
        def get(self, storage_key: str) -> bytes:
            raise FileNotFoundError(f"missing: {storage_key}")

        def put(self, storage_key: str, data: bytes, content_type: str) -> None:  # pragma: no cover
            raise NotImplementedError

        def delete(self, storage_key: str) -> None:  # pragma: no cover
            raise NotImplementedError

        def exists(self, storage_key: str) -> bool:  # pragma: no cover
            raise NotImplementedError

        def signed_url(self, storage_key: str, expires_seconds: int) -> str:  # pragma: no cover
            raise NotImplementedError

    await run_pipeline_for_job(
        job_id=seeded_job["job_id"], settings=settings, engine=engine, storage=_BrokenStorage()
    )

    job = await _fetch_job(engine, seeded_job["job_id"])
    assert job["status"] == "failed"
    assert job["error_code"] == "pipeline_error"
    assert "missing" in job["error_message"]
