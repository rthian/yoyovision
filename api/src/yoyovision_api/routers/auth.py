"""Minimal dev-only login endpoint. See `yoyovision_api.auth` module docstring."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from yoyovision_api.auth import AuthError, authenticate_user, create_access_token
from yoyovision_api.deps import DbSession, SettingsDep
from yoyovision_api.schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, session: DbSession, settings: SettingsDep) -> TokenResponse:
    try:
        user = await authenticate_user(session, payload.email, payload.password)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    token = create_access_token(user.id, settings)
    return TokenResponse(access_token=token)
