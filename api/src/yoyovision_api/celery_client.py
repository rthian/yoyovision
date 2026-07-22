"""Lightweight Celery client used only to enqueue tasks by name.

Deliberately does NOT import the `yoyovision_workers` package: the API and
workers are independently deployable services that only share the Postgres
schema (via `yoyovision_api.db_models`) and the Redis broker contract (task
name + kwargs). Enqueuing by name keeps that boundary real, not just
theoretical -- the API can never accidentally start executing a task body
in-process.
"""

from __future__ import annotations

from celery import Celery

from yoyovision_api.config import Settings

#: Must match the task name workers register in `yoyovision_workers.tasks`.
RUN_ANALYSIS_PIPELINE_TASK_NAME = "yoyovision_workers.tasks.run_analysis_pipeline_task"

_celery_client: Celery | None = None


def get_celery_client(settings: Settings) -> Celery:
    global _celery_client
    if _celery_client is None:
        _celery_client = Celery(
            "yoyovision_api_client",
            broker=settings.celery_broker_url,
            backend=settings.celery_result_backend,
        )
    return _celery_client


def enqueue_analysis_pipeline(settings: Settings, job_id: str) -> str:
    """Enqueues the pipeline task on the CPU queue; returns the Celery task id."""
    client = get_celery_client(settings)
    async_result = client.send_task(
        RUN_ANALYSIS_PIPELINE_TASK_NAME,
        kwargs={"job_id": job_id},
        queue="cpu",
    )
    return str(async_result.id)
