"""Health/readiness checks for the workers service.

Prompt F: "Add health and readiness checks." Exposed two ways: `tasks.py`
registers a Celery task wrapping `check_worker_health` (a liveness probe --
if a task can run at all, the broker path and this worker process are
alive), and any operator tooling can call `check_worker_health` directly
against a real or test engine.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from yoyovision_ml.inference.model_registry import get_model_registry

from yoyovision_workers.config import Settings
from yoyovision_workers.db import get_engine, reset_engine


async def check_worker_health(settings: Settings, engine: AsyncEngine | None = None) -> dict[str, Any]:
    """Checks Postgres connectivity and reports the process-wide model
    registry's loaded-model summary (path-free -- see
    `ModelRegistry.describe`, Prompt F: "never expose local model paths").

    `health_check_task` drives this with its own `asyncio.run(...)` per
    Celery task, exactly like `pipeline_runner.run_pipeline_for_job` -- so
    an engine obtained from `get_engine()` here must be disposed *and*
    dropped from the cache before returning, or the next task's (new event
    loop's) use of the same cached engine intermittently raises
    `RuntimeError: Event loop is closed` (see `db.reset_engine`'s
    docstring). Callers that pass their own `engine` (tests) own its
    lifecycle themselves.
    """
    owns_engine = engine is None
    engine = engine or get_engine(settings)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        database_status = "ok"
    except Exception as exc:  # noqa: BLE001 - readiness check must never raise
        database_status = f"unreachable: {exc}"
    finally:
        if owns_engine:
            await engine.dispose()
            reset_engine()

    return {
        "status": "ok" if database_status == "ok" else "degraded",
        "database": database_status,
        "loaded_models": get_model_registry().describe(),
    }
