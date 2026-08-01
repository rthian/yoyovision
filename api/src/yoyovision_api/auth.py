"""Minimal dev-only JWT authentication.

This is explicitly NOT production-grade SSO/OAuth. It exists so `owner_id`
and ownership checks are meaningful in this MVP. See README for the
documented limitation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from yoyovision_api.config import Settings
from yoyovision_api.db_models import User
from yoyovision_api.judging_enums import UserRole

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthError(Exception):
    """Raised for any authentication failure (bad credentials, bad/expired token)."""


def hash_password(plain_password: str) -> str:
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return _pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject_user_id: str, settings: Settings) -> str:
    expire_at = datetime.now(UTC) + timedelta(minutes=settings.auth_jwt_expire_minutes)
    payload = {"sub": subject_user_id, "exp": expire_at}
    return cast(
        str, jwt.encode(payload, settings.auth_jwt_secret, algorithm=settings.auth_jwt_algorithm)
    )


def decode_access_token(token: str, settings: Settings) -> str:
    try:
        payload = jwt.decode(
            token, settings.auth_jwt_secret, algorithms=[settings.auth_jwt_algorithm]
        )
    except JWTError as exc:
        raise AuthError("Invalid or expired access token.") from exc
    subject = payload.get("sub")
    if not isinstance(subject, str):
        raise AuthError("Access token missing subject.")
    return subject


async def authenticate_user(session: AsyncSession, email: str, password: str) -> User:
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(password, user.hashed_password):
        raise AuthError("Invalid email or password.")
    return user


async def ensure_dev_seed_user(session: AsyncSession, settings: Settings) -> User:
    """Creates the dev seed user (from .env config) if it does not exist yet.
    Only meaningful in development; production deployments should replace
    this auth module with a real identity provider."""
    result = await session.execute(
        select(User).where(User.email == settings.auth_dev_seed_user_email)
    )
    user = result.scalar_one_or_none()
    if user is not None:
        if user.role != UserRole.ADMIN:
            user.role = UserRole.ADMIN
            await session.commit()
            await session.refresh(user)
        return user
    user = User(
        email=settings.auth_dev_seed_user_email,
        hashed_password=hash_password(settings.auth_dev_seed_user_password),
        role=UserRole.ADMIN,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user
