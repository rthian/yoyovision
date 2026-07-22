"""Core, framework-agnostic orchestration for one analysis job: load the job
+ video row, run the `yoyovision_ml` pipeline (mock adapters by default,
real adapters wherever `Settings`/`adapter_kwargs` name them), and persist
events/deductions/score -- or a structured failure -- back to Postgres via
the Core tables in `schema.py`.

Deliberately separated from `tasks.py` (the thin Celery entrypoint) so this
async function can be exercised directly in tests without a Celery worker,
broker, or real Postgres/S3 (see `workers/tests/test_pipeline_runner.py`).

Prompt F (production inference) additions:

* Runs `run_analysis_pipeline` in a worker thread (`asyncio.to_thread`) so a
  concurrent coroutine can keep polling `analysis_jobs.cancel_requested`
  from Postgres and flip a shared `threading.Event` the pipeline's
  `CancellationToken` checks between stages -- see `_poll_cancel_requested`.
* Classifies failures via the `TransientPipelineError`/`DeterministicPipelineError`
  taxonomy: transient failures increment `retry_count` and re-raise so
  `tasks.py`'s Celery `autoretry_for` retries the job; deterministic
  failures (including explicit cancellation) are recorded once and never
  retried.
* Persistence of events/deductions/score is delete-then-insert per job, so a
  retried attempt (or a manual re-trigger) never leaves duplicate rows from
  a prior partial run -- idempotent by job id.
* Persists `model_versions`/`device`/`runtime_versions`/`stage_durations_ms`
  on the job row, and best-effort writes a human-readable `report.md` and a
  machine-readable `result.json` through the storage abstraction.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Event as ThreadingEvent
from typing import Any

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine
from yoyovision_ml.domain import (
    AnalysisEventPrediction,
    DeductionPrediction,
    JobStatus,
    PipelineStage,
    ScoreBreakdown,
)
from yoyovision_ml.inference.cancellation import CancellationToken
from yoyovision_ml.inference.device import DeviceInfo
from yoyovision_ml.inference.errors import (
    DeterministicPipelineError,
    PipelineCancelledError,
    TransientPipelineError,
)
from yoyovision_ml.inference.model_registry import get_model_registry
from yoyovision_ml.inference.report import generate_human_readable_report
from yoyovision_ml.inference.timing import StageTimings
from yoyovision_ml.interfaces import StoragePort
from yoyovision_ml.pipeline import PipelineResult, StageCallback, run_analysis_pipeline
from yoyovision_ml.ruleset import Ruleset, default_ruleset, get_ruleset_by_version

from yoyovision_workers.config import Settings
from yoyovision_workers.db import get_engine, reset_engine
from yoyovision_workers.logging_setup import get_logger
from yoyovision_workers.schema import (
    analysis_events,
    analysis_jobs,
    major_deductions,
    score_breakdowns,
    video_assets,
)
from yoyovision_workers.storage_factory import build_storage

logger = get_logger(__name__)

_DEFAULT_FPS_IF_UNKNOWN = 30.0

#: Rough progress fraction to record when each stage *finishes* -- purely
#: cosmetic (drives a progress bar), never used for correctness.
_STAGE_PROGRESS: dict[PipelineStage, float] = {
    PipelineStage.POSE_EXTRACTION: 0.15,
    PipelineStage.HAND_EXTRACTION: 0.30,
    PipelineStage.YOYO_DETECTION: 0.45,
    PipelineStage.TRACKING: 0.55,
    PipelineStage.STRING_ANALYSIS: 0.65,
    PipelineStage.FEATURE_EXTRACTION: 0.75,
    PipelineStage.TEMPORAL_EVENT_DETECTION: 0.85,
    PipelineStage.SCORING: 0.92,
}


def _event_values(job_id: str, event: AnalysisEventPrediction, now: datetime) -> dict[str, Any]:
    evidence_json = {"evidence": [asdict(ref) for ref in event.evidence]}
    return {
        "id": str(uuid.uuid4()),
        "analysis_id": job_id,
        "label": event.label,
        "family": event.family.value,
        "start_ms": event.start_ms,
        "end_ms": event.end_ms,
        "confidence": event.confidence,
        "outcome": event.outcome.value,
        "difficulty_band": event.difficulty_band.value,
        "source": "model",
        "review_status": "pending",
        "model_name": event.model_name,
        "model_version": event.model_version,
        "evidence_json": evidence_json,
        "created_at": now,
        "updated_at": now,
    }


def _deduction_values(
    job_id: str, deduction: DeductionPrediction, ruleset: Ruleset
) -> dict[str, Any]:
    """`points` mirrors the same versioned ruleset rate the scoring engine
    used to compute `ScoreBreakdown.major_deductions`, so the persisted row
    is consistent with the score it contributed to (and stays a fully
    editable, auditable value a human can override during review)."""
    rule = ruleset.deduction_rule_for(deduction.type)
    points = (rule.points_per_occurrence * deduction.quantity) if rule is not None else 0.0
    return {
        "id": str(uuid.uuid4()),
        "analysis_id": job_id,
        "type": deduction.type.value,
        "timestamp_ms": deduction.timestamp_ms,
        "quantity": deduction.quantity,
        "points": points,
        "confidence": deduction.confidence,
        "source": "model",
        "review_status": "pending",
    }


def _score_values(job_id: str, score: ScoreBreakdown, now: datetime) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "analysis_id": job_id,
        "technical_raw": score.technical_raw,
        "technical_scaled": score.technical_scaled,
        "freestyle_evaluation_raw": score.freestyle_evaluation_raw,
        "freestyle_evaluation_scaled": score.freestyle_evaluation_scaled,
        "major_deductions": score.major_deductions,
        "final_score": score.final_score,
        "confidence": score.confidence,
        "ruleset_version": score.ruleset_version,
        "warnings": score.warnings,
        "created_at": now,
    }


async def run_pipeline_for_job(
    job_id: str,
    settings: Settings,
    engine: AsyncEngine | None = None,
    storage: StoragePort | None = None,
) -> None:
    """Runs the full pipeline for `job_id` and persists the outcome.

    Any failure (missing rows, storage errors, pipeline exceptions) is caught
    and recorded on the job row as `status=failed` with a machine-readable
    `error_code` -- this function never raises for expected failure modes, so
    a Celery retry/backoff policy can be layered on top by `tasks.py` without
    this function needing to know about Celery at all.

    `tasks.py` drives this coroutine with one `asyncio.run(...)` call per
    Celery task, which gives each task its own event loop. `get_engine()`
    below returns a single process-wide `AsyncEngine`, so if we didn't
    dispose its pool *and* drop the cache before returning, a second task
    landing on the same forked worker process would either reuse asyncpg
    connections opened on the first task's now-closed event loop, or reuse
    the disposed engine's still-loop-bound internal pool primitives -- both
    surface as `RuntimeError: ... attached to a different loop` / "Event
    loop is closed" (see `db.reset_engine`'s docstring for why `dispose()`
    alone isn't sufficient). Callers that pass their own `engine` (tests)
    opt out of this auto-dispose/reset, since they own that engine's
    lifecycle themselves.
    """
    owns_engine = engine is None
    engine = engine or get_engine(settings)
    storage = storage or build_storage(settings)
    job_logger = logger.bind(job_id=job_id)

    try:
        await _run_pipeline_for_job(job_id, settings, engine, storage, job_logger)
    finally:
        if owns_engine:
            await engine.dispose()
            reset_engine()


async def _run_pipeline_for_job(
    job_id: str,
    settings: Settings,
    engine: AsyncEngine,
    storage: StoragePort,
    job_logger: Any,
) -> None:
    async with engine.begin() as conn:
        job_row = (
            (await conn.execute(select(analysis_jobs).where(analysis_jobs.c.id == job_id)))
            .mappings()
            .first()
        )
        if job_row is None:
            job_logger.warning("analysis_job_not_found")
            return

        video_row = (
            (
                await conn.execute(
                    select(video_assets).where(video_assets.c.id == job_row["video_id"])
                )
            )
            .mappings()
            .first()
        )
        if video_row is None:
            await _mark_failed(
                conn, job_id, "video_not_found", "Referenced video no longer exists."
            )
            job_logger.error("video_asset_not_found", video_id=job_row["video_id"])
            return

        await conn.execute(
            update(analysis_jobs)
            .where(analysis_jobs.c.id == job_id)
            .values(
                status=JobStatus.RUNNING.value,
                current_stage=PipelineStage.PREPROCESSING.value,
                progress=0.05,
                started_at=datetime.now(UTC),
            )
        )

    job_logger = job_logger.bind(video_id=video_row["id"])
    job_logger.info("pipeline_started", is_shadow=bool(job_row["is_shadow"]))

    tmp_path: Path | None = None
    try:
        video_bytes = storage.get(video_row["storage_key"])
        suffix = Path(video_row["storage_key"]).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = Path(tmp_file.name)
    except Exception as exc:  # noqa: BLE001 - storage failure before the pipeline even starts
        job_logger.error("pipeline_failed", error=str(exc))
        async with engine.begin() as conn:
            await _mark_failed(conn, job_id, "pipeline_error", str(exc)[:2048])
        return

    ruleset = get_ruleset_by_version(settings.ruleset_version) or default_ruleset()

    #: Cooperative cancellation across the async/sync boundary: the pipeline
    #: itself is a plain blocking function, so it runs on a worker thread
    #: (`asyncio.to_thread`) while this coroutine keeps polling Postgres and
    #: flips `cancel_event` -- a `threading.Event`, safe to set from this
    #: coroutine and read from the pipeline's thread -- the instant a human
    #: requests cancellation.
    cancel_event = ThreadingEvent()
    if job_row["cancel_requested"]:
        cancel_event.set()
    cancellation = CancellationToken(cancel_check=cancel_event.is_set, timeout_s=settings.pipeline_timeout_s)
    loop = asyncio.get_running_loop()
    stage_callback = _make_stage_callback(loop, engine, job_id, job_logger)
    poll_task = asyncio.create_task(
        _poll_cancel_requested(engine, job_id, cancel_event, settings.pipeline_cancel_poll_interval_s)
    )

    try:
        result = await asyncio.to_thread(
            run_analysis_pipeline,
            video_path=tmp_path,
            duration_ms=video_row["duration_ms"] or 0,
            fps=video_row["fps"] or _DEFAULT_FPS_IF_UNKNOWN,
            ruleset=ruleset,
            sample_fps=settings.pipeline_sample_fps,
            device_preference=settings.pipeline_device,
            model_registry=get_model_registry(),
            cancellation=cancellation,
            stage_callback=stage_callback,
        )
    except PipelineCancelledError as exc:
        job_logger.warning("pipeline_cancelled")
        async with engine.begin() as conn:
            await _mark_failed(
                conn, job_id, "cancelled", str(exc)[:2048] or "Cancelled by request.",
                status=JobStatus.CANCELLED,
            )
        return
    except TransientPipelineError as exc:
        # Retryable: record the attempt and re-raise so `tasks.py`'s
        # `autoretry_for=(TransientPipelineError,)` schedules another try.
        job_logger.warning("pipeline_transient_failure", error=str(exc))
        async with engine.begin() as conn:
            await _mark_transient_failure(conn, job_id, "pipeline_transient_error", str(exc)[:2048])
        raise
    except Exception as exc:  # noqa: BLE001 - DeterministicPipelineError + any unexpected failure
        job_logger.error("pipeline_failed", error=str(exc))
        async with engine.begin() as conn:
            await _mark_failed(conn, job_id, "pipeline_error", str(exc)[:2048])
        return
    finally:
        poll_task.cancel()
        with suppress(asyncio.CancelledError):
            await poll_task
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    now = datetime.now(UTC)
    async with engine.begin() as conn:
        await conn.execute(
            update(analysis_jobs)
            .where(analysis_jobs.c.id == job_id)
            .values(current_stage=PipelineStage.SCORING.value, progress=0.9)
        )
        # Idempotent persistence: clear anything a previous (retried or
        # manually re-triggered) attempt already wrote for this job before
        # inserting the fresh set, so retries never produce duplicate rows.
        await conn.execute(delete(analysis_events).where(analysis_events.c.analysis_id == job_id))
        await conn.execute(delete(major_deductions).where(major_deductions.c.analysis_id == job_id))
        await conn.execute(delete(score_breakdowns).where(score_breakdowns.c.analysis_id == job_id))

        if result.events:
            await conn.execute(
                insert(analysis_events), [_event_values(job_id, e, now) for e in result.events]
            )
        if result.deductions:
            await conn.execute(
                insert(major_deductions),
                [_deduction_values(job_id, d, ruleset) for d in result.deductions],
            )
        await conn.execute(
            insert(score_breakdowns).values(**_score_values(job_id, result.score, now))
        )

        await conn.execute(
            update(analysis_jobs)
            .where(analysis_jobs.c.id == job_id)
            .values(
                status=JobStatus.COMPLETED.value,
                progress=1.0,
                current_stage=PipelineStage.DONE.value,
                completed_at=now,
                error_code=None,
                error_message=None,
                model_versions=result.model_versions,
                device=result.device,
                runtime_versions=result.runtime_versions,
                stage_durations_ms=result.stage_durations_ms,
            )
        )

    job_logger.info(
        "pipeline_completed",
        event_count=len(result.events),
        deduction_count=len(result.deductions),
        model_versions=result.model_versions,
        device=result.device,
        is_shadow=bool(job_row["is_shadow"]),
    )

    _store_artifacts(storage, job_id, video_row, settings, result, job_logger)


def _make_stage_callback(
    loop: asyncio.AbstractEventLoop, engine: AsyncEngine, job_id: str, job_logger: Any
) -> StageCallback:
    """Builds the `stage_callback` passed into `run_analysis_pipeline`.

    That callback runs on the pipeline's worker thread (via
    `asyncio.to_thread`), not this coroutine's event loop thread, so it
    cannot `await` directly -- `asyncio.run_coroutine_threadsafe` schedules
    the actual DB write back onto `loop` instead. Progress updates are
    best-effort: a failure here must never turn a successfully-running
    pipeline into a failed job.
    """

    def _callback(stage: PipelineStage, elapsed_ms: float) -> None:
        progress = _STAGE_PROGRESS.get(stage, 0.5)

        async def _persist() -> None:
            try:
                async with engine.begin() as conn:
                    await conn.execute(
                        update(analysis_jobs)
                        .where(analysis_jobs.c.id == job_id)
                        .values(current_stage=stage.value, progress=progress)
                    )
            except Exception as exc:  # noqa: BLE001 - progress updates are best-effort
                job_logger.warning(
                    "stage_progress_persist_failed", stage=stage.value, error=str(exc)
                )

        asyncio.run_coroutine_threadsafe(_persist(), loop)

    return _callback


async def _poll_cancel_requested(
    engine: AsyncEngine, job_id: str, cancel_event: ThreadingEvent, interval_s: float
) -> None:
    """Re-reads `analysis_jobs.cancel_requested` every `interval_s` while a
    job is running; sets `cancel_event` the first time it observes `true` so
    the pipeline's `CancellationToken.check(...)` raises at the next stage
    boundary. Cancelled by the caller once the pipeline run finishes."""
    while True:
        await asyncio.sleep(interval_s)
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    select(analysis_jobs.c.cancel_requested).where(analysis_jobs.c.id == job_id)
                )
            ).first()
        if row is not None and row[0]:
            cancel_event.set()
            return


def _store_artifacts(
    storage: StoragePort,
    job_id: str,
    video_row: Mapping[str, Any],
    settings: Settings,
    result: PipelineResult,
    job_logger: Any,
) -> None:
    """Writes a human-readable `report.md` and machine-readable
    `result.json` for this job through the storage abstraction. Best-effort:
    a storage failure here is logged, never allowed to turn an already
    successfully-scored job into a failed one."""
    try:
        report_md = generate_human_readable_report(
            job_id=job_id,
            video_filename=Path(video_row["storage_key"]).name,
            pipeline_version=settings.pipeline_version,
            result=result,
            timings=StageTimings(durations_ms=dict(result.stage_durations_ms)),
            device_info=DeviceInfo(
                requested=settings.pipeline_device, resolved=result.device, available=True, reason=""
            ),
            runtime_versions=result.runtime_versions,
            monitoring=result.monitoring,
        )
        storage.put(f"analyses/{job_id}/report.md", report_md.encode("utf-8"), "text/markdown")
        storage.put(f"analyses/{job_id}/result.json", _result_json_bytes(result), "application/json")
    except Exception as exc:  # noqa: BLE001 - artefact storage is best-effort
        job_logger.warning("artifact_storage_failed", error=str(exc))


def _result_json_bytes(result: PipelineResult) -> bytes:
    payload = {
        "model_versions": result.model_versions,
        "device": result.device,
        "runtime_versions": result.runtime_versions,
        "stage_durations_ms": result.stage_durations_ms,
        "score": asdict(result.score),
        "events": [asdict(e) for e in result.events],
        "deductions": [asdict(d) for d in result.deductions],
        "monitoring": asdict(result.monitoring) if result.monitoring is not None else None,
    }
    return json.dumps(payload, default=str, indent=2).encode("utf-8")


async def _mark_failed(
    conn: Any,
    job_id: str,
    error_code: str,
    error_message: str,
    status: JobStatus = JobStatus.FAILED,
) -> None:
    await conn.execute(
        update(analysis_jobs)
        .where(analysis_jobs.c.id == job_id)
        .values(
            status=status.value,
            error_code=error_code,
            error_message=error_message,
            completed_at=datetime.now(UTC),
        )
    )


async def _mark_transient_failure(
    conn: Any, job_id: str, error_code: str, error_message: str
) -> None:
    """Records a retryable failure without marking the job terminally
    `failed` -- Celery is about to retry it. `retry_count` is incremented via
    a SQL expression (`+ 1`) rather than read-then-write, so concurrent
    updates from the same job (there should only ever be one) can't race."""
    await conn.execute(
        update(analysis_jobs)
        .where(analysis_jobs.c.id == job_id)
        .values(
            status=JobStatus.PENDING.value,
            error_code=error_code,
            error_message=error_message,
            retry_count=analysis_jobs.c.retry_count + 1,
        )
    )


async def mark_job_retries_exhausted(
    job_id: str, settings: Settings, engine: AsyncEngine | None = None
) -> None:
    """Called by `tasks.py` once Celery's own `max_retries` has just been hit
    for a `TransientPipelineError` -- flips the job from `pending` (queued
    for a retry that will now never happen) to a terminal `failed` status so
    it never appears stuck forever.

    `engine` is only ever passed explicitly by tests (mirroring
    `run_pipeline_for_job`'s own pattern); the real caller (`tasks.py`) lets
    this dispose the process-wide engine's pool *and* drop the cache per
    task run, for the same cross-event-loop reason documented on
    `run_pipeline_for_job` / `db.reset_engine`.
    """
    owns_engine = engine is None
    engine = engine or get_engine(settings)
    try:
        async with engine.begin() as conn:
            await _mark_failed(
                conn,
                job_id,
                "retries_exhausted",
                "Pipeline failed after exhausting all retry attempts.",
            )
    finally:
        if owns_engine:
            await engine.dispose()
            reset_engine()
