from __future__ import annotations

from datetime import UTC, datetime, timedelta

from yoyovision_api.services.invite_token import (
    TOKEN_TTL,
    generate_invite_token,
    hash_token,
    is_token_active,
    token_expires_at,
)


def test_generate_invite_token_returns_hashable_secret() -> None:
    raw, token_hash, prefix = generate_invite_token()
    assert raw
    assert token_hash == hash_token(raw)
    assert raw.startswith(prefix)


def test_token_expires_in_two_days() -> None:
    issued = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    expires = token_expires_at(issued_at=issued)
    assert expires - issued == TOKEN_TTL


def test_is_token_active_rejects_expired_or_revoked() -> None:
    now = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
    expires = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)
    assert not is_token_active(token_expires_at=expires, revoked_at=None, now=now)
    assert not is_token_active(
        token_expires_at=datetime(2026, 8, 5, tzinfo=UTC),
        revoked_at=datetime(2026, 8, 1, tzinfo=UTC),
        now=now,
    )
    assert is_token_active(
        token_expires_at=datetime(2026, 8, 5, tzinfo=UTC),
        revoked_at=None,
        now=now,
    )
