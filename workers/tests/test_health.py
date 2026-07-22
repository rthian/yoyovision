"""Tests for `yoyovision_workers.health.check_worker_health` (Prompt F health
and readiness checks): reports Postgres reachability and never raises, even
when the database connection is unusable."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from yoyovision_workers.config import Settings
from yoyovision_workers.health import check_worker_health


async def test_check_worker_health_reports_ok_when_database_is_reachable(
    engine: AsyncEngine, settings: Settings
) -> None:
    result = await check_worker_health(settings, engine=engine)

    assert result["status"] == "ok"
    assert result["database"] == "ok"
    assert isinstance(result["loaded_models"], dict)


async def test_check_worker_health_reports_degraded_when_database_is_unreachable(
    settings: Settings,
) -> None:
    broken_engine = create_async_engine("sqlite+aiosqlite:///no/such/directory/db.sqlite3")

    result = await check_worker_health(settings, engine=broken_engine)

    assert result["status"] == "degraded"
    assert "unreachable" in result["database"]

    await broken_engine.dispose()
