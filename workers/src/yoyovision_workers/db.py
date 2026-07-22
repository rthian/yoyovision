"""Async SQLAlchemy engine/connection helpers for the workers service.

Uses SQLAlchemy Core (`AsyncEngine.begin()` / `AsyncConnection`), not the ORM
session API, since `schema.py` only defines Core `Table` objects -- see that
module's docstring for why the worker doesn't share ORM classes with the API.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from yoyovision_workers.config import Settings

_engine: AsyncEngine | None = None


def build_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True, future=True)


def get_engine(settings: Settings) -> AsyncEngine:
    """Returns the process-wide engine, building it from `settings` on first
    use. Tests that need an isolated engine (e.g. in-memory SQLite) should
    build their own with `build_engine` and pass it explicitly rather than
    going through this cache.

    Every caller that obtains an engine through this cache (rather than
    passing its own) MUST call `reset_engine()` after `await engine.dispose()`
    once it's done -- see that function's docstring for why disposing alone
    is not enough across `asyncio.run()` boundaries."""
    global _engine
    if _engine is None:
        _engine = build_engine(settings.database_url)
    return _engine


def reset_engine() -> None:
    """Drops the cached process-wide engine so the next `get_engine()` call
    builds a brand new one.

    Each Celery task runs its own `asyncio.run(...)`, giving it a fresh event
    loop. `AsyncEngine.dispose()` closes pooled connections, but the
    underlying `AsyncAdaptedQueuePool` also holds asyncio primitives (locks,
    queues, futures) created against whichever loop was running the first
    time the pool was used -- `dispose()` does not recreate those. Reusing
    the *same* (disposed) `AsyncEngine` object from a later task, on a new
    event loop, intermittently raises `RuntimeError: Event loop is closed` /
    "Future ... attached to a different loop" as those stale primitives are
    touched from the new loop. Rebuilding the engine from scratch, rather
    than reusing a disposed one, avoids this: call this immediately after
    `await engine.dispose()` whenever the engine came from `get_engine()`."""
    global _engine
    _engine = None
