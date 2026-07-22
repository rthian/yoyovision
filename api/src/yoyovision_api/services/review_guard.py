"""Guards that block mutations on submitted analyses."""

from __future__ import annotations

from fastapi import HTTPException, status
from yoyovision_ml.domain import AnalysisReviewState, JobStatus

from yoyovision_api.db_models import AnalysisJobORM


def ensure_analysis_editable(job: AnalysisJobORM) -> None:
    if job.review_state == AnalysisReviewState.SUBMITTED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This analysis has been submitted and is locked for editing. "
                "Reopen it to make changes."
            ),
        )


def ensure_analysis_submittable(job: AnalysisJobORM) -> None:
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only completed analyses can be submitted for review.",
        )
    if job.review_state == AnalysisReviewState.SUBMITTED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This analysis has already been submitted.",
        )
