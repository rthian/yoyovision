"""CRUD + review actions for `AnalysisEvent` rows.

Implements product principle #4 ("Users must be able to add, edit, delete
and confirm every detected event") with a full audit trail: edits made by a
human always flip `source` to `human` and `review_status` to `edited`
(unless the caller explicitly sets a different `review_status`), and every
mutation triggers a score recomputation so the displayed score never goes
stale relative to the event list.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import select
from yoyovision_ml.domain import ReviewStatus, Source

from yoyovision_api.db_models import AnalysisEventORM
from yoyovision_api.deps import DbSession, OwnedJob, SettingsDep
from yoyovision_api.schemas import AnalysisEventCreate, AnalysisEventRead, AnalysisEventUpdate
from yoyovision_api.services.review_guard import ensure_analysis_editable
from yoyovision_api.services.scoring_service import recompute_score

router = APIRouter(prefix="/analyses/{analysis_id}/events", tags=["events"])


async def _get_owned_event(job: OwnedJob, event_id: str, session: DbSession) -> AnalysisEventORM:
    result = await session.execute(
        select(AnalysisEventORM).where(
            AnalysisEventORM.id == event_id, AnalysisEventORM.analysis_id == job.id
        )
    )
    event = result.scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found.")
    return event


@router.get("", response_model=list[AnalysisEventRead])
async def list_events(job: OwnedJob, session: DbSession) -> list[AnalysisEventORM]:
    result = await session.execute(
        select(AnalysisEventORM)
        .where(AnalysisEventORM.analysis_id == job.id)
        .order_by(AnalysisEventORM.start_ms)
    )
    return list(result.scalars().all())


@router.post("", response_model=AnalysisEventRead, status_code=status.HTTP_201_CREATED)
async def create_event(
    job: OwnedJob,
    payload: AnalysisEventCreate,
    session: DbSession,
    settings: SettingsDep,
) -> AnalysisEventORM:
    ensure_analysis_editable(job)
    event = AnalysisEventORM(
        analysis_id=job.id,
        label=payload.label,
        family=payload.family,
        start_ms=payload.start_ms,
        end_ms=payload.end_ms,
        confidence=payload.confidence,
        outcome=payload.outcome,
        difficulty_band=payload.difficulty_band,
        source=Source.HUMAN,
        review_status=ReviewStatus.CONFIRMED,
        model_name=None,
        model_version=None,
        evidence_json={"note": payload.notes} if payload.notes else {},
    )
    session.add(event)
    await session.flush()
    await recompute_score(session, job, settings.ruleset_version)
    await session.commit()
    return event


@router.patch("/{event_id}", response_model=AnalysisEventRead)
async def update_event(
    job: OwnedJob,
    event_id: str,
    payload: AnalysisEventUpdate,
    session: DbSession,
    settings: SettingsDep,
) -> AnalysisEventORM:
    ensure_analysis_editable(job)
    event = await _get_owned_event(job, event_id, session)

    changes = payload.model_dump(exclude_unset=True, exclude={"review_status"})
    for field_name, value in changes.items():
        setattr(event, field_name, value)

    if changes:
        event.source = Source.HUMAN
    event.review_status = payload.review_status or (
        ReviewStatus.EDITED if changes else event.review_status
    )

    await session.flush()
    await recompute_score(session, job, settings.ruleset_version)
    await session.commit()
    return event


@router.post("/{event_id}/confirm", response_model=AnalysisEventRead)
async def confirm_event(job: OwnedJob, event_id: str, session: DbSession) -> AnalysisEventORM:
    ensure_analysis_editable(job)
    event = await _get_owned_event(job, event_id, session)
    event.review_status = ReviewStatus.CONFIRMED
    await session.commit()
    return event


@router.post("/{event_id}/reject", response_model=AnalysisEventRead)
async def reject_event(
    job: OwnedJob, event_id: str, session: DbSession, settings: SettingsDep
) -> AnalysisEventORM:
    ensure_analysis_editable(job)
    event = await _get_owned_event(job, event_id, session)
    event.review_status = ReviewStatus.REJECTED
    await session.flush()
    await recompute_score(session, job, settings.ruleset_version)
    await session.commit()
    return event


@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_event(
    job: OwnedJob, event_id: str, session: DbSession, settings: SettingsDep
) -> Response:
    ensure_analysis_editable(job)
    event = await _get_owned_event(job, event_id, session)
    await session.delete(event)
    await session.flush()
    await recompute_score(session, job, settings.ruleset_version)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
