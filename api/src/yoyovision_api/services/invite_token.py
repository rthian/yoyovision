"""Private judge invite token generation and verification."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

TOKEN_TTL = timedelta(days=2)


def generate_invite_token() -> tuple[str, str, str]:
    """Returns (raw_token, sha256_hex_hash, display_prefix)."""
    raw_token = secrets.token_urlsafe(32)
    token_hash = hash_token(raw_token)
    token_prefix = raw_token[:8]
    return raw_token, token_hash, token_prefix


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def token_expires_at(*, issued_at: datetime | None = None) -> datetime:
    base = issued_at or datetime.now(UTC)
    return base + TOKEN_TTL


def is_token_active(
  *,
  token_expires_at: datetime,
  revoked_at: datetime | None,
  now: datetime | None = None,
) -> bool:
    current = now or datetime.now(UTC)
    if revoked_at is not None:
        return False
    expires = token_expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    return current <= expires
