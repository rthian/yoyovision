"""Shared pytest fixtures for the workers test suite.

Uses an in-memory SQLite engine (via `aiosqlite`) with the Core tables from
`yoyovision_workers.schema` created directly -- mirroring the approach in
`api/tests/conftest.py` -- so `pipeline_runner.run_pipeline_for_job` can be
exercised end to end without a real Postgres or S3/MinIO instance.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool
from yoyovision_ml.storage import LocalFilesystemStorage

from yoyovision_workers.config import Settings
from yoyovision_workers.schema import analysis_jobs, metadata, video_assets


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    test_engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with test_engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    try:
        yield test_engine
    finally:
        await test_engine.dispose()


@pytest.fixture
def settings(tmp_path: object) -> Settings:
    return Settings(storage_backend="local", storage_local_root=str(tmp_path))


@pytest.fixture
def storage(settings: Settings) -> LocalFilesystemStorage:
    return LocalFilesystemStorage(root=settings.storage_local_root)


@pytest_asyncio.fixture
async def seeded_job(engine: AsyncEngine, storage: LocalFilesystemStorage) -> dict[str, str]:
    """Inserts one `video_assets` row (with real bytes in `storage`) and one
    `pending` `analysis_jobs` row referencing it, as if the API had just
    handled an upload -- returns their ids."""
    video_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    storage_key = f"videos/owner-1/{video_id}.mp4"
    storage.put(storage_key, b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1024, "video/mp4")

    now = datetime.now(UTC)
    async with engine.begin() as conn:
        await conn.execute(
            insert(video_assets).values(
                id=video_id,
                owner_id="owner-1",
                original_filename="freestyle.mp4",
                storage_key=storage_key,
                mime_type="video/mp4",
                duration_ms=8_000,
                width=1280,
                height=720,
                fps=30.0,
                file_size=1024,
                status="ready",
                created_at=now,
                deleted_at=None,
            )
        )
        await conn.execute(
            insert(analysis_jobs).values(
                id=job_id,
                video_id=video_id,
                status="pending",
                progress=0.0,
                current_stage="queued",
                pipeline_version="0.1.0-dev",
                created_at=now,
            )
        )
    return {"video_id": video_id, "job_id": job_id}
