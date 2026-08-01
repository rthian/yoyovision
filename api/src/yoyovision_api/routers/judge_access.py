"""Token-authenticated judge access API (Phase C)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response

from yoyovision_api.deps import DbSession, SettingsDep, StorageDep
from yoyovision_api.judging_enums import JudgingEntryStatus
from yoyovision_api.schemas import (
    JudgeAccessRead,
    JudgeAccessVideoRead,
    JudgeFreestyleScoreRead,
    JudgeFreestyleScoreUpsert,
)
from yoyovision_api.services import judging_service
from yoyovision_api.services.judge_rate_limit import JudgeRateLimitExceeded, check_judge_rate_limit

router = APIRouter(prefix="/judge-access", tags=["judge-access"])


def _rate_limit_key(request: Request, token: str) -> str:
    client = request.client.host if request.client else "unknown"
    return f"{client}:{token[:8]}"


def _check_rate_limit(request: Request, token: str, settings: object) -> None:
    limit = min(getattr(settings, "api_rate_limit_per_minute", 60), 30)
    try:
        check_judge_rate_limit(_rate_limit_key(request, token), limit_per_minute=limit)
    except JudgeRateLimitExceeded as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc


def _map_service_error(exc: judging_service.JudgingServiceError) -> HTTPException:
    if isinstance(exc, judging_service.InviteInvalidError):
        return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))
    if isinstance(exc, judging_service.InviteInactiveError):
        return HTTPException(status_code=status.HTTP_410_GONE, detail=str(exc))
    if isinstance(exc, judging_service.JudgingAccessError):
        message = str(exc)
        if message == "Video not found.":
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
        if message in {"Scores already submitted.", "All freestyle fields are required to submit."}:
            code = (
                status.HTTP_409_CONFLICT
                if message == "Scores already submitted."
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            return HTTPException(status_code=code, detail=message)
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=message)
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


async def _resolve_assignment(
    session: DbSession, token: str, *, require_readable: bool
) -> object:
    assignment = await judging_service.resolve_assignment_by_token(session, token)
    if require_readable:
        judging_service._assert_entry_readable(assignment.entry)  # noqa: SLF001
    return assignment


def _score_to_read(score) -> JudgeFreestyleScoreRead | None:
    if score is None:
        return None
    return JudgeFreestyleScoreRead.model_validate(score)


def _build_access_read(assignment: object) -> JudgeAccessRead:
    entry = assignment.entry
    videos: list[JudgeAccessVideoRead] = []
    for entry_video in sorted(entry.videos, key=lambda row: row.sort_order):
        score = judging_service._score_for_video(assignment, entry_video.id)  # noqa: SLF001
        asset = entry_video.video
        videos.append(
            JudgeAccessVideoRead(
                entry_video_id=entry_video.id,
                sort_order=entry_video.sort_order,
                original_filename=asset.original_filename,
                duration_ms=asset.duration_ms,
                mime_type=asset.mime_type,
                my_score=_score_to_read(score),
            )
        )
    return JudgeAccessRead(
        assignment_id=assignment.id,
        display_name=assignment.display_name,
        entry_id=entry.id,
        entry_title=entry.title,
        entry_mode=entry.mode,
        entry_status=entry.status,
        due_at=entry.due_at,
        token_expires_at=assignment.token_expires_at,
        videos=videos,
    )


@router.get("/{token}", response_model=JudgeAccessRead)
async def get_judge_access(
    token: str,
    request: Request,
    session: DbSession,
    settings: SettingsDep,
) -> JudgeAccessRead:
    _check_rate_limit(request, token, settings)
    try:
        assignment = await _resolve_assignment(session, token, require_readable=True)
    except judging_service.JudgingServiceError as exc:
        raise _map_service_error(exc) from exc
    return _build_access_read(assignment)


@router.get("/{token}/videos/{entry_video_id}/stream")
async def stream_judge_video(
    token: str,
    entry_video_id: str,
    request: Request,
    session: DbSession,
    settings: SettingsDep,
    storage: StorageDep,
) -> Response:
    _check_rate_limit(request, token, settings)
    try:
        assignment = await _resolve_assignment(session, token, require_readable=True)
        entry_video = await judging_service.get_entry_video_for_assignment(
            assignment, entry_video_id
        )
        video = entry_video.video
        if video is None or video.deleted_at is not None:
            raise judging_service.JudgingAccessError("Video not found.")
    except judging_service.JudgingServiceError as exc:
        raise _map_service_error(exc) from exc

    data = storage.get(video.storage_key)
    return Response(content=data, media_type=video.mime_type)


@router.put("/{token}/videos/{entry_video_id}/fe", response_model=JudgeFreestyleScoreRead)
async def upsert_judge_fe(
    token: str,
    entry_video_id: str,
    payload: JudgeFreestyleScoreUpsert,
    request: Request,
    session: DbSession,
    settings: SettingsDep,
) -> JudgeFreestyleScoreRead:
    _check_rate_limit(request, token, settings)
    try:
        assignment = await judging_service.resolve_assignment_by_token(session, token)
        judging_service._assert_entry_writable(assignment.entry)  # noqa: SLF001
        score = await judging_service.upsert_judge_fe(
            session, assignment, entry_video_id, payload
        )
    except judging_service.JudgingServiceError as exc:
        raise _map_service_error(exc) from exc
    return JudgeFreestyleScoreRead.model_validate(score)


@router.post(
    "/{token}/videos/{entry_video_id}/submit",
    response_model=JudgeFreestyleScoreRead,
)
async def submit_judge_fe(
    token: str,
    entry_video_id: str,
    payload: JudgeFreestyleScoreUpsert,
    request: Request,
    session: DbSession,
    settings: SettingsDep,
) -> JudgeFreestyleScoreRead:
    _check_rate_limit(request, token, settings)
    try:
        assignment = await judging_service.resolve_assignment_by_token(session, token)
        score = await judging_service.submit_judge_fe(
            session, assignment, entry_video_id, payload
        )
    except judging_service.JudgingServiceError as exc:
        raise _map_service_error(exc) from exc
    return JudgeFreestyleScoreRead.model_validate(score)
