"""FastAPI application factory and startup wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Import for side-effect: registers the "mock" pose/hand/yoyo/tracker/temporal
# adapters with `yoyovision_ml.adapters_registry` (product principle #5/#7).
from yoyovision_ml import adapters_mock  # noqa: F401
from yoyovision_ml import storage as _storage_adapters  # noqa: F401

from yoyovision_api.auth import ensure_dev_seed_user
from yoyovision_api.config import get_settings
from yoyovision_api.db import AsyncSessionLocal, get_db_session
from yoyovision_api.logging_setup import configure_logging
from yoyovision_api.routers import (
    analyses,
    auth,
    deductions,
    evaluations,
    events,
    exports,
    judge_access,
    judging_entries,
    rulesets,
    videos,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.environment == "development":
        async with AsyncSessionLocal() as session:
            user = await ensure_dev_seed_user(session, settings)
            logger.info("dev_seed_user_ready", user_id=user.id, email=user.email)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="YoYoVision API",
        version="0.1.0",
        description=(
            "AI-assisted 1A yo-yo freestyle analysis platform. Training and "
            "judge-assistance tool only -- scores are never certified by "
            "IYYF, WYYC, or any competition body."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", path=str(request.url.path), error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Liveness probe: the process is up and serving requests. Never
        touches the database -- see `/health/ready` for that."""
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def readiness(
        session: Annotated[AsyncSession, Depends(get_db_session)],
    ) -> JSONResponse:
        """Readiness probe (Prompt F): confirms Postgres is actually
        reachable, not just that the process started. Returns 503 (not a
        raised exception) on failure so load balancers/orchestrators treat
        it as a normal, expected "not ready yet" signal. Uses the same
        `get_db_session` dependency as every other route (not the raw
        `AsyncSessionLocal`) so tests can override it like any other
        endpoint."""
        try:
            await session.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001 - readiness check must never crash the endpoint
            logger.warning("readiness_check_failed", error=str(exc))
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "database": "unreachable"},
            )
        return JSONResponse(content={"status": "ready", "database": "ok"})

    app.include_router(auth.router)
    app.include_router(videos.router)
    app.include_router(analyses.router)
    app.include_router(events.router)
    app.include_router(deductions.router)
    app.include_router(evaluations.router)
    app.include_router(exports.router)
    app.include_router(rulesets.router)
    app.include_router(judging_entries.router)
    app.include_router(judge_access.router)

    return app


app = create_app()
