"""Celery application: separate CPU and GPU task queues.

Per the spec's Workers requirements ("Separate CPU and GPU task queues"),
both queues are declared here as real Celery `Queue`s, and `-Q gpu` /
`-Q cpu` workers can be started independently (see `docker-compose.yml`'s
`worker-cpu` / `worker-gpu` services) even though only a CPU-routed task
(`run_analysis_pipeline_task`) exists today -- the mock adapters have no GPU
requirement. Swapping in a GPU-bound detector later means adding a task
routed to the `gpu` queue, not restructuring the broker topology.
"""

from __future__ import annotations

from celery import Celery
from kombu import Queue

from yoyovision_workers.config import get_settings
from yoyovision_workers.logging_setup import configure_logging

CPU_QUEUE = "cpu"
GPU_QUEUE = "gpu"


def build_celery_app() -> Celery:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = Celery(
        "yoyovision_workers",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["yoyovision_workers.tasks"],
    )
    app.conf.update(
        task_queues=(Queue(CPU_QUEUE), Queue(GPU_QUEUE)),
        task_default_queue=CPU_QUEUE,
        task_routes={
            "yoyovision_workers.tasks.run_analysis_pipeline_task": {"queue": CPU_QUEUE},
        },
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        result_expires=86_400,
    )
    return app


celery_app = build_celery_app()
