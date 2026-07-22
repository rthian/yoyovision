"""Tests for the Celery app/task wiring: queue topology and the task-name
contract that `yoyovision_api.celery_client` enqueues against by string name
(the two services never import each other's task/route modules)."""

from __future__ import annotations

from yoyovision_workers.celery_app import CPU_QUEUE, GPU_QUEUE, celery_app
from yoyovision_workers.tasks import run_analysis_pipeline_task

_EXPECTED_TASK_NAME = "yoyovision_workers.tasks.run_analysis_pipeline_task"


def test_pipeline_task_is_registered_under_the_expected_name() -> None:
    assert run_analysis_pipeline_task.name == _EXPECTED_TASK_NAME
    assert _EXPECTED_TASK_NAME in celery_app.tasks


def test_pipeline_task_is_routed_to_the_cpu_queue() -> None:
    route = celery_app.conf.task_routes[_EXPECTED_TASK_NAME]
    assert route["queue"] == CPU_QUEUE


def test_both_cpu_and_gpu_queues_are_declared() -> None:
    queue_names = {queue.name for queue in celery_app.conf.task_queues}
    assert queue_names == {CPU_QUEUE, GPU_QUEUE}
