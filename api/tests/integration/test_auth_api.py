"""Integration tests for the dev-only JWT login endpoint."""

from __future__ import annotations

from httpx import AsyncClient

from yoyovision_api.db_models import User


async def test_login_with_valid_credentials_returns_bearer_token(
    client: AsyncClient, test_user: User
) -> None:
    response = await client.post(
        "/auth/login", json={"email": test_user.email, "password": "correct-horse"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str) and body["access_token"]


async def test_login_with_wrong_password_returns_401(client: AsyncClient, test_user: User) -> None:
    response = await client.post(
        "/auth/login", json={"email": test_user.email, "password": "wrong-password"}
    )
    assert response.status_code == 401


async def test_login_with_unknown_email_returns_401(client: AsyncClient) -> None:
    response = await client.post(
        "/auth/login", json={"email": "nobody@yoyovision.local", "password": "irrelevant"}
    )
    assert response.status_code == 401


async def test_token_from_login_is_accepted_by_protected_endpoints(
    client: AsyncClient, test_user: User
) -> None:
    login_response = await client.post(
        "/auth/login", json={"email": test_user.email, "password": "correct-horse"}
    )
    token = login_response.json()["access_token"]

    response = await client.get("/videos", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json() == []


async def test_malformed_bearer_token_is_rejected(client: AsyncClient) -> None:
    response = await client.get("/videos", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401
