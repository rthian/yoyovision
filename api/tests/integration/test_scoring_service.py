"""Integration tests for `yoyovision_api.services.scoring_service.recompute_score`
against a real (in-memory SQLite) DB session, covering the human review
audit-trail behaviour: rejected events/deductions are excluded from scoring
but the rows themselves are preserved."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from yoyovision_ml.domain import (
    DeductionType,
    DifficultyBand,
    EventFamily,
    JobStatus,
    Outcome,
    ReviewStatus,
    Source,
    VideoStatus,
)

from yoyovision_api.db_models import (
    AnalysisEventORM,
    AnalysisJobORM,
    MajorDeductionORM,
    User,
    VideoAssetORM,
)
from yoyovision_api.services.scoring_service import recompute_score


async def _make_job(session: AsyncSession, owner: User) -> AnalysisJobORM:
    video = VideoAssetORM(
        owner_id=owner.id,
        original_filename="freestyle.mp4",
        storage_key=f"videos/{owner.id}/example.mp4",
        mime_type="video/mp4",
        duration_ms=20_000,
        width=1920,
        height=1080,
        fps=30.0,
        file_size=1_000_000,
        status=VideoStatus.READY,
    )
    session.add(video)
    await session.flush()

    job = AnalysisJobORM(
        video_id=video.id, status=JobStatus.COMPLETED, progress=1.0, pipeline_version="0.1.0-dev"
    )
    session.add(job)
    await session.flush()
    return job


async def test_recompute_score_counts_confirmed_success_events(
    db_session: AsyncSession, test_user: User
) -> None:
    job = await _make_job(db_session, test_user)
    event = AnalysisEventORM(
        analysis_id=job.id,
        label="mount_0",
        family=EventFamily.MOUNT,
        start_ms=0,
        end_ms=500,
        confidence=0.9,
        outcome=Outcome.SUCCESS,
        difficulty_band=DifficultyBand.BASIC,
        source=Source.MODEL,
        review_status=ReviewStatus.CONFIRMED,
        model_name="mock-temporal-event-detector",
        model_version="0.0.0-mock",
    )
    db_session.add(event)
    await db_session.commit()

    breakdown = await recompute_score(db_session, job, "1a-draft-0.1")

    assert breakdown.technical_raw == 1.0  # basic band == 1.0 point
    assert any("unofficial" in w.lower() for w in breakdown.warnings)


async def test_recompute_score_excludes_rejected_events(
    db_session: AsyncSession, test_user: User
) -> None:
    job = await _make_job(db_session, test_user)
    event = AnalysisEventORM(
        analysis_id=job.id,
        label="mount_0",
        family=EventFamily.MOUNT,
        start_ms=0,
        end_ms=500,
        confidence=0.9,
        outcome=Outcome.SUCCESS,
        difficulty_band=DifficultyBand.ADVANCED,
        source=Source.MODEL,
        review_status=ReviewStatus.REJECTED,
        model_name="mock-temporal-event-detector",
        model_version="0.0.0-mock",
    )
    db_session.add(event)
    await db_session.commit()

    breakdown = await recompute_score(db_session, job, "1a-draft-0.1")

    assert breakdown.technical_raw == 0.0


async def test_recompute_score_applies_deduction_points(
    db_session: AsyncSession, test_user: User
) -> None:
    job = await _make_job(db_session, test_user)
    deduction = MajorDeductionORM(
        analysis_id=job.id,
        type=DeductionType.YOYO_STOP,
        timestamp_ms=5_000,
        quantity=1,
        points=2.0,
        confidence=0.8,
        source=Source.MODEL,
        review_status=ReviewStatus.CONFIRMED,
    )
    db_session.add(deduction)
    await db_session.commit()

    breakdown = await recompute_score(db_session, job, "1a-draft-0.1")

    assert breakdown.major_deductions == 2.0  # 1 occurrence x 2.0 points_per_occurrence


async def test_recompute_score_gates_pending_dangerous_play_review(
    db_session: AsyncSession, test_user: User
) -> None:
    """Prompt D: 'Dangerous-play detection must never automatically
    disqualify a player. It must create a review flag.' A freshly-detected
    `dangerous_play_review` deduction is persisted (`PENDING`) but must
    contribute zero score impact until a human reviewer confirms it."""
    job = await _make_job(db_session, test_user)
    deduction = MajorDeductionORM(
        analysis_id=job.id,
        type=DeductionType.DANGEROUS_PLAY_REVIEW,
        timestamp_ms=5_000,
        quantity=1,
        points=5.0,
        confidence=0.7,
        source=Source.MODEL,
        review_status=ReviewStatus.PENDING,
    )
    db_session.add(deduction)
    await db_session.commit()

    breakdown = await recompute_score(db_session, job, "1a-draft-0.1")

    assert breakdown.major_deductions == 0.0


async def test_recompute_score_applies_confirmed_dangerous_play_review(
    db_session: AsyncSession, test_user: User
) -> None:
    """Once a human reviewer sets `review_status=CONFIRMED` on the same
    flag, it scores like any other major deduction."""
    job = await _make_job(db_session, test_user)
    deduction = MajorDeductionORM(
        analysis_id=job.id,
        type=DeductionType.DANGEROUS_PLAY_REVIEW,
        timestamp_ms=5_000,
        quantity=1,
        points=5.0,
        confidence=0.7,
        source=Source.MODEL,
        review_status=ReviewStatus.CONFIRMED,
    )
    db_session.add(deduction)
    await db_session.commit()

    breakdown = await recompute_score(db_session, job, "1a-draft-0.1")

    assert breakdown.major_deductions == 5.0  # 1 occurrence x 5.0 points_per_occurrence
