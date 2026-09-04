"""Business logic for multi-judge video entries."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import inspect as sa_inspect, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from yoyovision_api.db_models import (
    AnalysisJobORM,
    JudgeAssignmentORM,
    JudgeClickORM,
    JudgeFreestyleScoreORM,
    JudgingEntryORM,
    JudgingEntryVideoORM,
    User,
    VideoAssetORM,
)
from yoyovision_api.judging_enums import (
    AggregationMode,
    AiMixProfile,
    ClickMode,
    JudgeAssignmentStatus,
    JudgingEntryMode,
    JudgingEntryStatus,
)
from yoyovision_api.schemas import JudgeClickCreate, JudgeFreestyleScoreUpsert
from yoyovision_api.services.invite_token import (
    generate_invite_token,
    hash_token,
    is_token_active,
    token_expires_at,
)

FE_SCORE_FIELDS = (
    "execution",
    "control",
    "trick_diversity",
    "space_use_emphasis",
    "music_choreography",
    "music_construction",
    "body_control",
    "showmanship",
)


class JudgingServiceError(ValueError):
    """Raised when a judging entry operation is invalid."""


class InviteInvalidError(JudgingServiceError):
    """Unknown invite token."""


class InviteInactiveError(JudgingServiceError):
    """Expired or revoked invite token."""


class JudgingAccessError(JudgingServiceError):
    """Judge cannot access or modify this resource."""


def _assignment_status(assignment: JudgeAssignmentORM) -> JudgeAssignmentStatus:
    if assignment.revoked_at is not None:
        return JudgeAssignmentStatus.PENDING
    if "freestyle_scores" in sa_inspect(assignment).unloaded:
        return JudgeAssignmentStatus.PENDING
    scores = assignment.freestyle_scores
    if scores and all(score.is_submitted for score in scores):
        return JudgeAssignmentStatus.SUBMITTED
    if scores and any(
        score.execution is not None
        or score.control is not None
        or score.trick_diversity is not None
        or score.space_use_emphasis is not None
        or score.music_choreography is not None
        or score.music_construction is not None
        or score.body_control is not None
        or score.showmanship is not None
        or score.notes
        for score in scores
    ):
        return JudgeAssignmentStatus.IN_PROGRESS
    return JudgeAssignmentStatus.PENDING


def _assert_entry_readable(entry: JudgingEntryORM) -> None:
    if entry.status == JudgingEntryStatus.DRAFT:
        raise JudgingAccessError("This judging entry is not open yet.")


def _assert_entry_writable(entry: JudgingEntryORM) -> None:
    if entry.status == JudgingEntryStatus.DRAFT:
        raise JudgingAccessError("This judging entry is not open yet.")
    if entry.status == JudgingEntryStatus.LOCKED:
        raise JudgingAccessError("This judging entry is locked.")


def _score_for_video(
    assignment: JudgeAssignmentORM, entry_video_id: str
) -> JudgeFreestyleScoreORM | None:
    for score in assignment.freestyle_scores:
        if score.entry_video_id == entry_video_id:
            return score
    return None


def _apply_fe_payload(score: JudgeFreestyleScoreORM, payload: JudgeFreestyleScoreUpsert) -> None:
    for field in FE_SCORE_FIELDS:
        setattr(score, field, getattr(payload, field))
    score.notes = payload.notes


def _all_fe_fields_present(score: JudgeFreestyleScoreORM) -> bool:
    return all(getattr(score, field) is not None for field in FE_SCORE_FIELDS)


async def create_entry(
    session: AsyncSession,
    *,
    admin: User,
    title: str,
    mode: JudgingEntryMode,
    ruleset_version: str,
    ai_mix_profile: AiMixProfile,
    aggregation_mode: AggregationMode,
    click_mode: ClickMode = ClickMode.OFF,
    due_at: datetime | None,
    video_ids: list[str],
) -> JudgingEntryORM:
    if not video_ids:
        raise JudgingServiceError("At least one video is required.")

    for video_id in video_ids:
        result = await session.execute(select(VideoAssetORM).where(VideoAssetORM.id == video_id))
        video = result.scalar_one_or_none()
        if video is None or video.deleted_at is not None:
            raise JudgingServiceError(f"Video not found: {video_id}")

    entry = JudgingEntryORM(
        title=title,
        mode=mode,
        status=JudgingEntryStatus.DRAFT,
        ruleset_version=ruleset_version,
        ai_mix_profile=ai_mix_profile,
        aggregation_mode=aggregation_mode,
        click_mode=click_mode,
        created_by=admin.id,
        due_at=due_at,
    )
    session.add(entry)
    await session.flush()

    for index, video_id in enumerate(video_ids):
        session.add(
            JudgingEntryVideoORM(entry_id=entry.id, video_id=video_id, sort_order=index)
        )

    await session.commit()
    return await get_entry(session, entry.id)


async def list_entries(session: AsyncSession) -> list[JudgingEntryORM]:
    result = await session.execute(
        select(JudgingEntryORM)
        .order_by(JudgingEntryORM.created_at.desc())
        .options(
            selectinload(JudgingEntryORM.videos).selectinload(JudgingEntryVideoORM.video),
            selectinload(JudgingEntryORM.judges).selectinload(JudgeAssignmentORM.freestyle_scores),
            selectinload(JudgingEntryORM.judges).selectinload(JudgeAssignmentORM.clicks),
        )
    )
    return list(result.scalars().all())


async def get_entry(session: AsyncSession, entry_id: str) -> JudgingEntryORM:
    result = await session.execute(
        select(JudgingEntryORM)
        .where(JudgingEntryORM.id == entry_id)
        .options(
            selectinload(JudgingEntryORM.videos).selectinload(JudgingEntryVideoORM.video),
            selectinload(JudgingEntryORM.judges).selectinload(JudgeAssignmentORM.freestyle_scores),
            selectinload(JudgingEntryORM.judges).selectinload(JudgeAssignmentORM.clicks),
        )
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        raise JudgingServiceError("Judging entry not found.")
    return entry


async def update_entry(
    session: AsyncSession,
    entry: JudgingEntryORM,
    *,
    title: str | None = None,
    mode: JudgingEntryMode | None = None,
    status: JudgingEntryStatus | None = None,
    ruleset_version: str | None = None,
    ai_mix_profile: AiMixProfile | None = None,
    aggregation_mode: AggregationMode | None = None,
    click_mode: ClickMode | None = None,
    due_at: datetime | None = None,
    clear_due_at: bool = False,
) -> JudgingEntryORM:
    if entry.status == JudgingEntryStatus.LOCKED and status != JudgingEntryStatus.LOCKED:
        if any(
            value is not None
            for value in (
                title,
                mode,
                ruleset_version,
                ai_mix_profile,
                aggregation_mode,
                click_mode,
            )
        ) or clear_due_at or due_at is not None:
            raise JudgingServiceError("Locked entries cannot be edited.")

    if title is not None:
        entry.title = title
    if mode is not None:
        entry.mode = mode
    if status is not None:
        entry.status = status
    if ruleset_version is not None:
        entry.ruleset_version = ruleset_version
    if ai_mix_profile is not None:
        entry.ai_mix_profile = ai_mix_profile
    if aggregation_mode is not None:
        entry.aggregation_mode = aggregation_mode
    if click_mode is not None:
        entry.click_mode = click_mode
    if clear_due_at:
        entry.due_at = None
    elif due_at is not None:
        entry.due_at = due_at

    await session.commit()
    return await get_entry(session, entry.id)


async def add_videos(
    session: AsyncSession,
    entry: JudgingEntryORM,
    video_ids: list[str],
) -> JudgingEntryORM:
    if entry.status == JudgingEntryStatus.LOCKED:
        raise JudgingServiceError("Locked entries cannot be modified.")
    if not video_ids:
        raise JudgingServiceError("At least one video id is required.")

    existing = {video.video_id for video in entry.videos}
    next_order = max((video.sort_order for video in entry.videos), default=-1) + 1

    for video_id in video_ids:
        if video_id in existing:
            continue
        result = await session.execute(select(VideoAssetORM).where(VideoAssetORM.id == video_id))
        video = result.scalar_one_or_none()
        if video is None or video.deleted_at is not None:
            raise JudgingServiceError(f"Video not found: {video_id}")
        session.add(
            JudgingEntryVideoORM(entry_id=entry.id, video_id=video_id, sort_order=next_order)
        )
        next_order += 1

    await session.commit()
    return await get_entry(session, entry.id)


async def link_entry_video_analyses(
    session: AsyncSession,
    entry_video: JudgingEntryVideoORM,
    *,
    official_analysis_id: str | None = None,
    shadow_analysis_id: str | None = None,
) -> JudgingEntryVideoORM:
    if official_analysis_id is not None:
        await _validate_analysis_for_video(session, official_analysis_id, entry_video.video_id)
        entry_video.official_analysis_id = official_analysis_id
    if shadow_analysis_id is not None:
        await _validate_analysis_for_video(session, shadow_analysis_id, entry_video.video_id)
        entry_video.shadow_analysis_id = shadow_analysis_id
    await session.commit()
    return entry_video


async def _validate_analysis_for_video(
    session: AsyncSession, analysis_id: str, video_id: str
) -> None:
    result = await session.execute(
        select(AnalysisJobORM).where(AnalysisJobORM.id == analysis_id)
    )
    job = result.scalar_one_or_none()
    if job is None or job.video_id != video_id:
        raise JudgingServiceError("Analysis does not belong to this video.")


async def add_judge(
    session: AsyncSession,
    entry: JudgingEntryORM,
    *,
    display_name: str,
    include_in_results: bool = True,
    is_shadow: bool = False,
) -> tuple[JudgeAssignmentORM, str]:
    if entry.status == JudgingEntryStatus.LOCKED:
        raise JudgingServiceError("Locked entries cannot accept new judges.")

    raw_token, token_hash, token_prefix = generate_invite_token()
    assignment = JudgeAssignmentORM(
        entry_id=entry.id,
        display_name=display_name.strip(),
        invite_token_hash=token_hash,
        token_prefix=token_prefix,
        token_expires_at=token_expires_at(),
        include_in_results=include_in_results,
        is_shadow=is_shadow,
    )
    session.add(assignment)
    await session.commit()
    await session.refresh(assignment)
    return assignment, raw_token


async def rotate_judge_token(
    session: AsyncSession, assignment: JudgeAssignmentORM
) -> str:
    raw_token, token_hash, token_prefix = generate_invite_token()
    assignment.invite_token_hash = token_hash
    assignment.token_prefix = token_prefix
    assignment.token_expires_at = token_expires_at()
    assignment.revoked_at = None
    await session.commit()
    return raw_token


async def revoke_judge(session: AsyncSession, assignment: JudgeAssignmentORM) -> None:
    assignment.revoked_at = datetime.now(UTC)
    await session.commit()


async def resolve_assignment_by_token(
    session: AsyncSession, raw_token: str
) -> JudgeAssignmentORM:
    token_hash = hash_token(raw_token)
    result = await session.execute(
        select(JudgeAssignmentORM)
        .where(JudgeAssignmentORM.invite_token_hash == token_hash)
        .options(
            selectinload(JudgeAssignmentORM.entry)
            .selectinload(JudgingEntryORM.videos)
            .selectinload(JudgingEntryVideoORM.video),
            selectinload(JudgeAssignmentORM.freestyle_scores),
            selectinload(JudgeAssignmentORM.clicks),
        )
    )
    assignment = result.scalar_one_or_none()
    if assignment is None:
        raise InviteInvalidError("Invalid invite link.")
    if not is_token_active(
        token_expires_at=assignment.token_expires_at,
        revoked_at=assignment.revoked_at,
    ):
        raise InviteInactiveError(
            "Invite link has expired or been revoked. Ask your admin for a new link."
        )
    return assignment


async def get_entry_video_for_assignment(
    assignment: JudgeAssignmentORM, entry_video_id: str
) -> JudgingEntryVideoORM:
    for entry_video in assignment.entry.videos:
        if entry_video.id == entry_video_id:
            return entry_video
    raise JudgingAccessError("Video not found.")


async def upsert_judge_fe(
    session: AsyncSession,
    assignment: JudgeAssignmentORM,
    entry_video_id: str,
    payload: JudgeFreestyleScoreUpsert,
) -> JudgeFreestyleScoreORM:
    _assert_entry_writable(assignment.entry)
    await get_entry_video_for_assignment(assignment, entry_video_id)

    score = _score_for_video(assignment, entry_video_id)
    if score is not None and score.is_submitted:
        raise JudgingAccessError("Scores already submitted.")

    if score is None:
        score = JudgeFreestyleScoreORM(
            assignment_id=assignment.id,
            entry_video_id=entry_video_id,
        )
        session.add(score)
        assignment.freestyle_scores.append(score)

    _apply_fe_payload(score, payload)
    await session.commit()
    await session.refresh(score)
    return score


async def submit_judge_fe(
    session: AsyncSession,
    assignment: JudgeAssignmentORM,
    entry_video_id: str,
    payload: JudgeFreestyleScoreUpsert,
) -> JudgeFreestyleScoreORM:
    _assert_entry_writable(assignment.entry)
    await get_entry_video_for_assignment(assignment, entry_video_id)

    score = _score_for_video(assignment, entry_video_id)
    if score is not None and score.is_submitted:
        raise JudgingAccessError("Scores already submitted.")

    if score is None:
        score = JudgeFreestyleScoreORM(
            assignment_id=assignment.id,
            entry_video_id=entry_video_id,
        )
        session.add(score)
        assignment.freestyle_scores.append(score)

    _apply_fe_payload(score, payload)
    if not _all_fe_fields_present(score):
        raise JudgingAccessError("All freestyle fields are required to submit.")

    score.is_submitted = True
    score.submitted_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(score)
    return score



def _clicks_for_video(
    assignment: JudgeAssignmentORM, entry_video_id: str
) -> list[JudgeClickORM]:
    return [click for click in assignment.clicks if click.entry_video_id == entry_video_id]


def _assert_clicks_editable(assignment: JudgeAssignmentORM, entry_video_id: str) -> None:
    _assert_entry_writable(assignment.entry)
    score = _score_for_video(assignment, entry_video_id)
    if score is not None and score.is_submitted:
        raise JudgingAccessError("Scores already submitted; clicks are locked.")


async def add_judge_click(
    session: AsyncSession,
    assignment: JudgeAssignmentORM,
    entry_video_id: str,
    payload: JudgeClickCreate,
) -> JudgeClickORM:
    if assignment.entry.click_mode == ClickMode.OFF:
        raise JudgingAccessError("Clicker is not enabled for this entry.")
    _assert_clicks_editable(assignment, entry_video_id)
    await get_entry_video_for_assignment(assignment, entry_video_id)

    click = JudgeClickORM(
        assignment_id=assignment.id,
        entry_video_id=entry_video_id,
        timestamp_ms=payload.timestamp_ms,
        label=payload.label,
    )
    session.add(click)
    assignment.clicks.append(click)
    await session.commit()
    await session.refresh(click)
    return click


async def delete_judge_click(
    session: AsyncSession,
    assignment: JudgeAssignmentORM,
    click_id: str,
) -> None:
    if assignment.entry.click_mode == ClickMode.OFF:
        raise JudgingAccessError("Clicker is not enabled for this entry.")
    click = next((row for row in assignment.clicks if row.id == click_id), None)
    if click is None:
        raise JudgingAccessError("Click not found.")
    _assert_clicks_editable(assignment, click.entry_video_id)
    await session.delete(click)
    await session.commit()

def build_invite_url(base_url: str, raw_token: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/{raw_token}"


def build_share_message(entry_title: str, judge_name: str, invite_url: str) -> str:
    return (
        f'You have been invited to judge "{entry_title}" as {judge_name}. '
        f"Open your private link (expires in 48 hours): {invite_url}"
    )


def assignment_status(assignment: JudgeAssignmentORM) -> JudgeAssignmentStatus:
    return _assignment_status(assignment)
