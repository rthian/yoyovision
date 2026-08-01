"""Admin API for multi-judge video entries."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from yoyovision_api.config import Settings
from yoyovision_api.db_models import JudgingEntryORM, JudgingEntryVideoORM, JudgeAssignmentORM
from yoyovision_api.deps import CurrentAdmin, DbSession, SettingsDep
from yoyovision_api.judging_enums import JudgingEntryMode
from yoyovision_api.schemas import (
    JudgeAssignmentCreate,
    JudgeAssignmentSummary,
    JudgeInviteRead,
    JudgingEntryCreate,
    JudgingEntryRead,
    JudgingEntryUpdate,
    JudgingEntryVideoAnalysisLink,
    JudgingEntryVideoAttach,
    JudgingEntryVideoRead,
    JudgingEntryResultsRead,
)
from yoyovision_api.services import judging_results_service, judging_service

router = APIRouter(prefix="/judging-entries", tags=["judging-entries"])


def _entry_to_read(entry: JudgingEntryORM) -> JudgingEntryRead:
    return JudgingEntryRead(
        id=entry.id,
        title=entry.title,
        mode=entry.mode,
        status=entry.status,
        ruleset_version=entry.ruleset_version,
        ai_mix_profile=entry.ai_mix_profile,
        aggregation_mode=entry.aggregation_mode,
        due_at=entry.due_at,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        videos=[
            JudgingEntryVideoRead(
                id=video.id,
                video_id=video.video_id,
                sort_order=video.sort_order,
                original_filename=video.video.original_filename,
                official_analysis_id=video.official_analysis_id,
                shadow_analysis_id=video.shadow_analysis_id,
            )
            for video in sorted(entry.videos, key=lambda row: row.sort_order)
        ],
        judges=[
            JudgeAssignmentSummary(
                id=judge.id,
                display_name=judge.display_name,
                token_prefix=judge.token_prefix,
                token_expires_at=judge.token_expires_at,
                include_in_results=judge.include_in_results,
                is_shadow=judge.is_shadow,
                revoked_at=judge.revoked_at,
                status=judging_service.assignment_status(judge),
            )
            for judge in entry.judges
        ],
    )


def _invite_read(
    settings: Settings,
    entry: JudgingEntryORM,
    assignment: JudgeAssignmentORM,
    raw_token: str,
) -> JudgeInviteRead:
    invite_url = judging_service.build_invite_url(settings.judge_invite_base_url, raw_token)
    return JudgeInviteRead(
        assignment_id=assignment.id,
        display_name=assignment.display_name,
        token_prefix=assignment.token_prefix,
        invite_url=invite_url,
        share_message=judging_service.build_share_message(
            entry.title, assignment.display_name, invite_url
        ),
        token_expires_at=assignment.token_expires_at,
        include_in_results=assignment.include_in_results,
        is_shadow=assignment.is_shadow,
        status=judging_service.assignment_status(assignment),
    )


@router.post("", response_model=JudgingEntryRead, status_code=status.HTTP_201_CREATED)
async def create_judging_entry(
    payload: JudgingEntryCreate,
    session: DbSession,
    admin: CurrentAdmin,
    settings: SettingsDep,
) -> JudgingEntryRead:
  try:
      entry = await judging_service.create_entry(
          session,
          admin=admin,
          title=payload.title,
          mode=payload.mode,
          ruleset_version=payload.ruleset_version or settings.ruleset_version,
          ai_mix_profile=payload.ai_mix_profile,
          aggregation_mode=payload.aggregation_mode,
          due_at=payload.due_at,
          video_ids=payload.video_ids,
      )
  except judging_service.JudgingServiceError as exc:
      raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
  return _entry_to_read(entry)


@router.get("", response_model=list[JudgingEntryRead])
async def list_judging_entries(session: DbSession, admin: CurrentAdmin) -> list[JudgingEntryRead]:
    entries = await judging_service.list_entries(session)
    return [_entry_to_read(entry) for entry in entries]


@router.get("/{entry_id}", response_model=JudgingEntryRead)
async def get_judging_entry(
    entry_id: str, session: DbSession, admin: CurrentAdmin
) -> JudgingEntryRead:
    try:
        entry = await judging_service.get_entry(session, entry_id)
    except judging_service.JudgingServiceError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _entry_to_read(entry)




@router.get("/{entry_id}/results", response_model=JudgingEntryResultsRead)
async def get_judging_entry_results(
    entry_id: str,
    session: DbSession,
    admin: CurrentAdmin,
) -> JudgingEntryResultsRead:
    try:
        return await judging_results_service.compute_entry_results(session, entry_id)
    except judging_service.JudgingServiceError as exc:
        status_code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in str(exc).lower()
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

@router.patch("/{entry_id}", response_model=JudgingEntryRead)
async def update_judging_entry(
    entry_id: str,
    payload: JudgingEntryUpdate,
    session: DbSession,
    admin: CurrentAdmin,
) -> JudgingEntryRead:
    try:
        entry = await judging_service.get_entry(session, entry_id)
        entry = await judging_service.update_entry(
            session,
            entry,
            title=payload.title,
            mode=payload.mode,
            status=payload.status,
            ruleset_version=payload.ruleset_version,
            ai_mix_profile=payload.ai_mix_profile,
            aggregation_mode=payload.aggregation_mode,
            due_at=payload.due_at,
            clear_due_at=payload.clear_due_at,
        )
    except judging_service.JudgingServiceError as exc:
        status_code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in str(exc).lower()
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return _entry_to_read(entry)


@router.post("/{entry_id}/videos", response_model=JudgingEntryRead)
async def attach_videos(
    entry_id: str,
    payload: JudgingEntryVideoAttach,
    session: DbSession,
    admin: CurrentAdmin,
) -> JudgingEntryRead:
    try:
        entry = await judging_service.get_entry(session, entry_id)
        entry = await judging_service.add_videos(session, entry, payload.video_ids)
    except judging_service.JudgingServiceError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return _entry_to_read(entry)


@router.patch(
    "/{entry_id}/videos/{entry_video_id}/analyses",
    response_model=JudgingEntryRead,
)
async def link_video_analyses(
    entry_id: str,
    entry_video_id: str,
    payload: JudgingEntryVideoAnalysisLink,
    session: DbSession,
    admin: CurrentAdmin,
) -> JudgingEntryRead:
    try:
        entry = await judging_service.get_entry(session, entry_id)
        entry_video = next((row for row in entry.videos if row.id == entry_video_id), None)
        if entry_video is None:
            raise judging_service.JudgingServiceError("Entry video not found.")
        await judging_service.link_entry_video_analyses(
            session,
            entry_video,
            official_analysis_id=payload.official_analysis_id,
            shadow_analysis_id=payload.shadow_analysis_id,
        )
        entry = await judging_service.get_entry(session, entry_id)
    except judging_service.JudgingServiceError as exc:
        status_code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in str(exc).lower()
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return _entry_to_read(entry)


@router.post("/{entry_id}/judges", response_model=JudgeInviteRead, status_code=status.HTTP_201_CREATED)
async def add_judge(
    entry_id: str,
    payload: JudgeAssignmentCreate,
    session: DbSession,
    admin: CurrentAdmin,
    settings: SettingsDep,
) -> JudgeInviteRead:
    try:
        entry = await judging_service.get_entry(session, entry_id)
        assignment, raw_token = await judging_service.add_judge(
            session,
            entry,
            display_name=payload.display_name,
            include_in_results=payload.include_in_results,
            is_shadow=payload.is_shadow,
        )
    except judging_service.JudgingServiceError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return _invite_read(settings, entry, assignment, raw_token)


@router.post("/{entry_id}/judges/{assignment_id}/rotate", response_model=JudgeInviteRead)
async def rotate_judge_invite(
    entry_id: str,
    assignment_id: str,
    session: DbSession,
    admin: CurrentAdmin,
    settings: SettingsDep,
) -> JudgeInviteRead:
    try:
        entry = await judging_service.get_entry(session, entry_id)
        assignment = await _get_assignment(session, entry_id, assignment_id)
        raw_token = await judging_service.rotate_judge_token(session, assignment)
    except judging_service.JudgingServiceError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _invite_read(settings, entry, assignment, raw_token)


@router.post("/{entry_id}/judges/{assignment_id}/revoke", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_judge_invite(
    entry_id: str,
    assignment_id: str,
    session: DbSession,
    admin: CurrentAdmin,
) -> None:
    try:
        assignment = await _get_assignment(session, entry_id, assignment_id)
        await judging_service.revoke_judge(session, assignment)
    except judging_service.JudgingServiceError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


async def _get_assignment(
    session: DbSession, entry_id: str, assignment_id: str
) -> JudgeAssignmentORM:
    result = await session.execute(
        select(JudgeAssignmentORM).where(
            JudgeAssignmentORM.id == assignment_id,
            JudgeAssignmentORM.entry_id == entry_id,
        )
    )
    assignment = result.scalar_one_or_none()
    if assignment is None:
        raise judging_service.JudgingServiceError("Judge assignment not found.")
    return assignment
