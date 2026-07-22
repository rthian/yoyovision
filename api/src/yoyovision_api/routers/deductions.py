"""CRUD + review actions for `MajorDeduction` rows (yo-yo stop/change/detach, etc.)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import select
from yoyovision_ml.domain import ReviewStatus, Source

from yoyovision_api.db_models import MajorDeductionORM
from yoyovision_api.deps import DbSession, OwnedJob, SettingsDep
from yoyovision_api.schemas import MajorDeductionCreate, MajorDeductionRead, MajorDeductionUpdate
from yoyovision_api.services.review_guard import ensure_analysis_editable
from yoyovision_api.services.scoring_service import recompute_score

router = APIRouter(prefix="/analyses/{analysis_id}/deductions", tags=["deductions"])


async def _get_owned_deduction(
    job: OwnedJob, deduction_id: str, session: DbSession
) -> MajorDeductionORM:
    result = await session.execute(
        select(MajorDeductionORM).where(
            MajorDeductionORM.id == deduction_id, MajorDeductionORM.analysis_id == job.id
        )
    )
    deduction = result.scalar_one_or_none()
    if deduction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deduction not found.")
    return deduction


@router.get("", response_model=list[MajorDeductionRead])
async def list_deductions(job: OwnedJob, session: DbSession) -> list[MajorDeductionORM]:
    result = await session.execute(
        select(MajorDeductionORM)
        .where(MajorDeductionORM.analysis_id == job.id)
        .order_by(MajorDeductionORM.timestamp_ms)
    )
    return list(result.scalars().all())


@router.post("", response_model=MajorDeductionRead, status_code=status.HTTP_201_CREATED)
async def create_deduction(
    job: OwnedJob,
    payload: MajorDeductionCreate,
    session: DbSession,
    settings: SettingsDep,
) -> MajorDeductionORM:
    ensure_analysis_editable(job)
    deduction = MajorDeductionORM(
        analysis_id=job.id,
        type=payload.type,
        timestamp_ms=payload.timestamp_ms,
        quantity=payload.quantity,
        points=payload.points,
        confidence=1.0,
        source=Source.HUMAN,
        review_status=ReviewStatus.CONFIRMED,
    )
    session.add(deduction)
    await session.flush()
    await recompute_score(session, job, settings.ruleset_version)
    await session.commit()
    return deduction


@router.patch("/{deduction_id}", response_model=MajorDeductionRead)
async def update_deduction(
    job: OwnedJob,
    deduction_id: str,
    payload: MajorDeductionUpdate,
    session: DbSession,
    settings: SettingsDep,
) -> MajorDeductionORM:
    ensure_analysis_editable(job)
    deduction = await _get_owned_deduction(job, deduction_id, session)

    changes = payload.model_dump(exclude_unset=True, exclude={"review_status"})
    for field_name, value in changes.items():
        setattr(deduction, field_name, value)
    if changes:
        deduction.source = Source.HUMAN
    deduction.review_status = payload.review_status or (
        ReviewStatus.EDITED if changes else deduction.review_status
    )

    await session.flush()
    await recompute_score(session, job, settings.ruleset_version)
    await session.commit()
    return deduction


@router.post("/{deduction_id}/confirm", response_model=MajorDeductionRead)
async def confirm_deduction(
    job: OwnedJob, deduction_id: str, session: DbSession
) -> MajorDeductionORM:
    ensure_analysis_editable(job)
    deduction = await _get_owned_deduction(job, deduction_id, session)
    deduction.review_status = ReviewStatus.CONFIRMED
    await session.commit()
    return deduction


@router.post("/{deduction_id}/reject", response_model=MajorDeductionRead)
async def reject_deduction(
    job: OwnedJob, deduction_id: str, session: DbSession, settings: SettingsDep
) -> MajorDeductionORM:
    ensure_analysis_editable(job)
    deduction = await _get_owned_deduction(job, deduction_id, session)
    deduction.review_status = ReviewStatus.REJECTED
    await session.flush()
    await recompute_score(session, job, settings.ruleset_version)
    await session.commit()
    return deduction


@router.delete("/{deduction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_deduction(
    job: OwnedJob, deduction_id: str, session: DbSession, settings: SettingsDep
) -> Response:
    ensure_analysis_editable(job)
    deduction = await _get_owned_deduction(job, deduction_id, session)
    await session.delete(deduction)
    await session.flush()
    await recompute_score(session, job, settings.ruleset_version)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
