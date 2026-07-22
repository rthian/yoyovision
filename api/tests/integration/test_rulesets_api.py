"""Integration tests for the read-only ruleset transparency endpoints
(product principle #8: keep the rule set versioned and configurable, and
never hide the config that produced a score)."""

from __future__ import annotations

from httpx import AsyncClient


async def test_list_rulesets_includes_the_packaged_1a_draft(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get("/rulesets", headers=auth_headers)
    assert response.status_code == 200
    versions = [r["version"] for r in response.json()]
    assert "1a-draft-0.1" in versions
    draft = next(r for r in response.json() if r["version"] == "1a-draft-0.1")
    assert draft["is_official"] is False
    assert "not certified by IYYF" in draft["disclaimer"]


async def test_get_ruleset_by_version_returns_full_config(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get("/rulesets/1a-draft-0.1", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["difficulty_band_points"]["basic"] == 1.0
    assert any(rule["type"] == "yoyo_stop" for rule in body["deduction_rules"])


async def test_get_unknown_ruleset_version_returns_404(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get("/rulesets/does-not-exist", headers=auth_headers)
    assert response.status_code == 404


async def test_rulesets_endpoint_requires_authentication(client: AsyncClient) -> None:
    response = await client.get("/rulesets")
    assert response.status_code == 401


async def test_list_rulesets_includes_iyyf_wyyc_25_draft(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get("/rulesets", headers=auth_headers)
    assert response.status_code == 200
    versions = [r["version"] for r in response.json()]
    assert "iyyf-wyyc-25-draft" in versions
    draft = next(r for r in response.json() if r["version"] == "iyyf-wyyc-25-draft")
    assert draft["is_official"] is False
    assert draft["technical_weight"] == 0.6
    assert draft["freestyle_evaluation_weight"] == 0.4
