"""Manual Freestyle Evaluation entry (MVP scope: placeholders + manual values only)."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select
from yoyovision_ml.domain import Source

from yoyovision_api.db_models import FreestyleEvaluationORM
from yoyovision_api.deps import DbSession, OwnedJob, SettingsDep
from yoyovision_api.schemas import FreestyleEvaluationRead, FreestyleEvaluationUpsert
from yoyovision_api.services.scoring_service import recompute_score

router = APIRouter(prefix="/analyses/{analysis_id}/evaluation", tags=["evaluations"])


@router.get("", response_model=FreestyleEvaluationRead | None)
async def get_evaluation(job: OwnedJob, session: DbSession) -> FreestyleEvaluationORM | None:
    result = await session.execute(
        select(FreestyleEvaluationORM).where(FreestyleEvaluationORM.analysis_id == job.id)
    )
    return result.scalar_one_or_none()


@router.put("", response_model=FreestyleEvaluationRead)
async def upsert_evaluation(
    job: OwnedJob,
    payload: FreestyleEvaluationUpsert,
    session: DbSession,
    settings: SettingsDep,
) -> FreestyleEvaluationORM:
    result = await session.execute(
        select(FreestyleEvaluationORM).where(FreestyleEvaluationORM.analysis_id == job.id)
    )
    evaluation = result.scalar_one_or_none()
    if evaluation is None:
        evaluation = FreestyleEvaluationORM(analysis_id=job.id)
        session.add(evaluation)

    evaluation.execution = payload.execution
    evaluation.control = payload.control
    evaluation.trick_diversity = payload.trick_diversity
    evaluation.space_use_emphasis = payload.space_use_emphasis
    evaluation.music_choreography = payload.music_choreography
    evaluation.music_construction = payload.music_construction
    evaluation.body_control = payload.body_control
    evaluation.showmanship = payload.showmanship
    evaluation.source = Source.HUMAN
    evaluation.notes = payload.notes

    await session.flush()
    await recompute_score(session, job, settings.ruleset_version)
    await session.commit()
    return evaluation
