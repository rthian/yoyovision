"""Prompt F (production inference) tests for `pipeline_runner`: idempotent
persistence, cancellation, transient-vs-deterministic retry classification,
Prompt F metadata persistence, and best-effort artefact storage.

Uses the same in-memory SQLite + local-filesystem fixtures as
`test_pipeline_runner.py`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine
from yoyovision_ml.inference.errors import PipelineTimeoutError
from yoyovision_ml.storage import LocalFilesystemStorage

import yoyovision_workers.pipeline_runner as pipeline_runner_module
from yoyovision_workers.config import Settings
from yoyovision_workers.pipeline_runner import mark_job_retries_exhausted, run_pipeline_for_job
from yoyovision_workers.schema import analysis_events, analysis_jobs, score_breakdowns


async def _fetch_job(engine: AsyncEngine, job_id: str) -> dict[str, Any]:
    async with engine.begin() as conn:
        row = (
            (await conn.execute(select(analysis_jobs).where(analysis_jobs.c.id == job_id)))
            .mappings()
            .first()
        )
        assert row is not None
        return dict(row)


async def test_completed_job_persists_prompt_f_metadata(
    engine: AsyncEngine, settings: Settings, storage: LocalFilesystemStorage, seeded_job: dict[str, str]
) -> None:
    await run_pipeline_for_job(
        job_id=seeded_job["job_id"], settings=settings, engine=engine, storage=storage
    )

    job = await _fetch_job(engine, seeded_job["job_id"])
    assert job["device"] == settings.pipeline_device
    assert job["model_versions"]
    assert any("mock-" in v or "0.0.0-mock" in v for v in job["model_versions"].values())
    assert job["runtime_versions"]
    assert job["stage_durations_ms"]
    assert job["retry_count"] == 0
    assert job["error_code"] is None


async def test_completed_job_writes_report_and_result_artifacts(
    engine: AsyncEngine, settings: Settings, storage: LocalFilesystemStorage, seeded_job: dict[str, str]
) -> None:
    await run_pipeline_for_job(
        job_id=seeded_job["job_id"], settings=settings, engine=engine, storage=storage
    )

    assert storage.exists(f"analyses/{seeded_job['job_id']}/report.md")
    assert storage.exists(f"analyses/{seeded_job['job_id']}/result.json")
    report_bytes = storage.get(f"analyses/{seeded_job['job_id']}/report.md")
    assert b"YoYoVision Analysis Report" in report_bytes


async def test_run_pipeline_for_job_is_idempotent_and_clears_stale_rows(
    engine: AsyncEngine, settings: Settings, storage: LocalFilesystemStorage, seeded_job: dict[str, str]
) -> None:
    """Simulates a stale row left behind by a previous (e.g. crashed/retried)
    attempt at this same job id -- a fresh run must replace it, not add to it."""
    now = datetime.now(UTC)
    async with engine.begin() as conn:
        await conn.execute(
            insert(analysis_events).values(
                id="stale-event",
                analysis_id=seeded_job["job_id"],
                label="stale",
                family="mount",
                start_ms=0,
                end_ms=1,
                confidence=0.5,
                outcome="success",
                difficulty_band="basic",
                source="model",
                review_status="pending",
                evidence_json={},
                created_at=now,
                updated_at=now,
            )
        )
        await conn.execute(
            insert(score_breakdowns).values(
                id="stale-score",
                analysis_id=seeded_job["job_id"],
                technical_raw=0.0,
                technical_scaled=0.0,
                freestyle_evaluation_raw=0.0,
                freestyle_evaluation_scaled=0.0,
                major_deductions=0.0,
                final_score=0.0,
                confidence=0.0,
                ruleset_version="stale",
                warnings=[],
                created_at=now,
            )
        )

    await run_pipeline_for_job(
        job_id=seeded_job["job_id"], settings=settings, engine=engine, storage=storage
    )

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
        score = (
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

    assert all(e["id"] != "stale-event" for e in events)
    assert score is not None
    assert score["id"] != "stale-score"
    assert score["ruleset_version"] != "stale"


async def test_run_pipeline_for_job_honors_pre_set_cancel_requested(
    engine: AsyncEngine, settings: Settings, storage: LocalFilesystemStorage, seeded_job: dict[str, str]
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            update(analysis_jobs)
            .where(analysis_jobs.c.id == seeded_job["job_id"])
            .values(cancel_requested=True)
        )

    await run_pipeline_for_job(
        job_id=seeded_job["job_id"], settings=settings, engine=engine, storage=storage
    )

    job = await _fetch_job(engine, seeded_job["job_id"])
    assert job["status"] == "cancelled"
    assert job["error_code"] == "cancelled"


async def test_transient_pipeline_error_increments_retry_count_and_reraises(
    engine: AsyncEngine,
    settings: Settings,
    storage: LocalFilesystemStorage,
    seeded_job: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_timeout(*args: object, **kwargs: object) -> None:
        raise PipelineTimeoutError("simulated timeout")

    monkeypatch.setattr(pipeline_runner_module, "run_analysis_pipeline", _raise_timeout)

    with pytest.raises(PipelineTimeoutError):
        await run_pipeline_for_job(
            job_id=seeded_job["job_id"], settings=settings, engine=engine, storage=storage
        )

    job = await _fetch_job(engine, seeded_job["job_id"])
    # Retryable: not marked "failed" (Celery is expected to retry), but the
    # attempt is recorded.
    assert job["status"] == "pending"
    assert job["retry_count"] == 1
    assert job["error_code"] == "pipeline_transient_error"


async def test_mark_job_retries_exhausted_marks_job_failed_terminally(
    engine: AsyncEngine, settings: Settings, seeded_job: dict[str, str]
) -> None:
    await mark_job_retries_exhausted(job_id=seeded_job["job_id"], settings=settings, engine=engine)

    job = await _fetch_job(engine, seeded_job["job_id"])
    assert job["status"] == "failed"
    assert job["error_code"] == "retries_exhausted"
    assert job["completed_at"] is not None
