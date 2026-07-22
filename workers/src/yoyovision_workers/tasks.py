"""Celery task entrypoints.

Thin by design: each task just drives the async `pipeline_runner` functions
with `asyncio.run` (Celery tasks are plain sync callables) and lets Celery
own retry/failure bookkeeping at the broker level, while `pipeline_runner`
owns recording the failure on the `AnalysisJob` row itself so the API/user
sees a structured reason even if Celery retries are exhausted silently.

`RUN_ANALYSIS_PIPELINE_TASK_NAME` in `yoyovision_api.celery_client` MUST
match this task's registered name -- the API enqueues purely by name/kwargs,
never by importing this module (see that module's docstring).

Prompt F retry policy: `run_pipeline_for_job` raises `TransientPipelineError`
for retryable failures (storage/network blips, soft timeouts) and swallows
`DeterministicPipelineError`s itself (marking the job `failed` with no
retry). `autoretry_for` below wires Celery's automatic backoff-retry to that
one exception family -- nothing else is ever retried.
"""

from __future__ import annotations

import asyncio
from typing import Any

from yoyovision_ml.inference.errors import TransientPipelineError

from yoyovision_workers.celery_app import celery_app
from yoyovision_workers.config import get_settings
from yoyovision_workers.health import check_worker_health
from yoyovision_workers.logging_setup import get_logger
from yoyovision_workers.pipeline_runner import mark_job_retries_exhausted, run_pipeline_for_job

logger = get_logger(__name__)


@celery_app.task(  # type: ignore[untyped-decorator]
    bind=True,
    name="yoyovision_workers.tasks.run_analysis_pipeline_task",
    max_retries=2,
    default_retry_delay=30,
    autoretry_for=(TransientPipelineError,),
    retry_backoff=True,
    retry_backoff_max=300,
)
def run_analysis_pipeline_task(self: Any, job_id: str) -> str:
    """Runs the full offline analysis pipeline for `job_id` (product
    principle #10: analysis is always offline/asynchronous, never run
    in-process by the API)."""
    logger.info("run_analysis_pipeline_task_received", job_id=job_id, attempt=self.request.retries)
    try:
        asyncio.run(run_pipeline_for_job(job_id=job_id, settings=get_settings()))
    except TransientPipelineError:
        if self.request.retries >= self.max_retries:
            # Celery's autoretry_for is about to give up for good; record a
            # terminal `failed` status so the job never sits stuck "running".
            logger.error("run_analysis_pipeline_task_retries_exhausted", job_id=job_id)
            asyncio.run(mark_job_retries_exhausted(job_id=job_id, settings=get_settings()))
        raise
    return job_id


@celery_app.task(  # type: ignore[untyped-decorator]
    name="yoyovision_workers.tasks.health_check_task",
)
def health_check_task() -> dict[str, Any]:
    """Liveness/readiness probe: successfully executing at all proves the
    broker + this worker process are alive; the returned payload also
    reports Postgres reachability and which models are currently loaded."""
    return asyncio.run(check_worker_health(get_settings()))
