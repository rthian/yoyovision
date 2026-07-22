"""Integration tests for `/health` and `/health/ready` (Prompt F)."""

from __future__ import annotations

from httpx import AsyncClient


async def test_health_returns_ok_without_touching_the_database(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness_returns_ready_when_database_is_reachable(client: AsyncClient) -> None:
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": "ok"}
