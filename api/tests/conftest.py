"""Shared pytest fixtures for API unit + integration tests.

Tests run against an in-memory SQLite database (via `aiosqlite`) rather than
Postgres, so the full request/response/DB round trip can be exercised
without Docker. Alembic migration correctness against real Postgres is
documented as a separate manual/CI check (see `docs/architecture.md`) since
this sandbox has no Postgres available; tables here are created directly
from `Base.metadata` to keep the ORM model itself as the single source of
truth under test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from yoyovision_ml.storage import LocalFilesystemStorage

from yoyovision_api.auth import create_access_token, hash_password
from yoyovision_api.config import Settings, get_settings
from yoyovision_api.db import Base, get_db_session
from yoyovision_api.db_models import User
from yoyovision_api.judging_enums import UserRole
from yoyovision_api.deps import get_storage
from yoyovision_api.main import create_app


@pytest_asyncio.fixture
async def db_session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    try:
        yield session_factory
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with db_session_factory() as session:
        yield session


@pytest.fixture
def test_settings(tmp_path: object) -> Settings:
    return Settings(storage_backend="local", storage_local_root=str(tmp_path))


@pytest_asyncio.fixture
async def app(
    db_session_factory: async_sessionmaker[AsyncSession],
    test_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
):
    async def _override_get_db_session() -> AsyncIterator[AsyncSession]:
        async with db_session_factory() as session:
            yield session

    def _override_get_storage() -> LocalFilesystemStorage:
        return LocalFilesystemStorage(root=test_settings.storage_local_root)

    monkeypatch.setattr(
        "yoyovision_api.services.job_service.enqueue_analysis_pipeline",
        lambda settings, job_id: "test-task-id",
    )

    fastapi_app = create_app()
    fastapi_app.dependency_overrides[get_db_session] = _override_get_db_session
    fastapi_app.dependency_overrides[get_settings] = lambda: test_settings
    fastapi_app.dependency_overrides[get_storage] = _override_get_storage
    return fastapi_app


@pytest_asyncio.fixture
async def client(app: object) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    user = User(email="rider@yoyovision.local", hashed_password=hash_password("correct-horse"))
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user: User, test_settings: Settings) -> dict[str, str]:
    token = create_access_token(test_user.id, test_settings)
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def admin_user(db_session: AsyncSession) -> User:
    user = User(
        email="admin@yoyovision.local",
        hashed_password=hash_password("admin-pass"),
        role=UserRole.ADMIN,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
def admin_headers(admin_user: User, test_settings: Settings) -> dict[str, str]:
    token = create_access_token(admin_user.id, test_settings)
    return {"Authorization": f"Bearer {token}"}
