"""Admin calibration: judge clicks vs official analysis events (Phase F)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from yoyovision_ml.domain import AnalysisEventPrediction
from yoyovision_ml.scoring.calibration import event_count_agreement
from yoyovision_ml.scoring.judges import match_clicks_to_events
from yoyovision_ml.scoring.types import JudgeClick

from yoyovision_api.db_models import (
    AnalysisEventORM,
    JudgeAssignmentORM,
    JudgeClickORM,
    JudgingEntryORM,
    JudgingEntryVideoORM,
)
from yoyovision_api.judging_enums import ClickMode
from yoyovision_api.schemas import (
    ClickMatchRead,
    JudgeClickCalibrationRead,
    JudgingEntryCalibrationRead,
    VideoClickCalibrationRead,
)
from yoyovision_api.services.judging_service import JudgingServiceError

DEFAULT_TOLERANCE_MS = 1000


def _event_to_prediction(event: AnalysisEventORM) -> AnalysisEventPrediction:
    return AnalysisEventPrediction(
        label=event.label,
        family=event.family,
        start_ms=event.start_ms,
        end_ms=event.end_ms,
        confidence=event.confidence,
        outcome=event.outcome,
        difficulty_band=event.difficulty_band,
        model_name=event.model_name or "unknown",
        model_version=event.model_version or "unknown",
    )


def _click_to_ml(assignment_id: str, click: JudgeClickORM) -> JudgeClick:
    return JudgeClick(
        click_id=click.id,
        judge_id=assignment_id,
        timestamp_ms=click.timestamp_ms,
        associated_label=click.label,
        notes="",
    )


async def _load_events(
    session: AsyncSession, analysis_id: str | None
) -> list[AnalysisEventORM]:
    if analysis_id is None:
        return []
    result = await session.execute(
        select(AnalysisEventORM).where(AnalysisEventORM.analysis_id == analysis_id)
    )
    return list(result.scalars().all())


async def compute_entry_calibration(
    session: AsyncSession,
    entry_id: str,
    *,
    tolerance_ms: int = DEFAULT_TOLERANCE_MS,
) -> JudgingEntryCalibrationRead:
    result = await session.execute(
        select(JudgingEntryORM)
        .where(JudgingEntryORM.id == entry_id)
        .options(
            selectinload(JudgingEntryORM.videos).selectinload(JudgingEntryVideoORM.video),
            selectinload(JudgingEntryORM.judges).selectinload(JudgeAssignmentORM.clicks),
        )
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        raise JudgingServiceError("Judging entry not found.")

    warnings: list[str] = []
    if entry.click_mode == ClickMode.OFF:
        warnings.append("Clicker is disabled for this entry (click_mode=off).")

    video_reads: list[VideoClickCalibrationRead] = []
    for entry_video in sorted(entry.videos, key=lambda row: row.sort_order):
        events_orm = await _load_events(session, entry_video.official_analysis_id)
        events = [_event_to_prediction(event) for event in events_orm]
        if entry_video.official_analysis_id and not events:
            warnings.append(
                f"No events on official analysis for {entry_video.video.original_filename}."
            )

        judge_reads: list[JudgeClickCalibrationRead] = []
        panel_counts: list[int] = []
        for assignment in entry.judges:
            video_clicks = [
                click
                for click in assignment.clicks
                if click.entry_video_id == entry_video.id
            ]
            ml_clicks = [_click_to_ml(assignment.id, click) for click in video_clicks]
            matches = match_clicks_to_events(ml_clicks, events, tolerance_ms=tolerance_ms)
            agreement = event_count_agreement(events, ml_clicks, tolerance_ms=tolerance_ms)
            if assignment.include_in_results and not assignment.is_shadow:
                panel_counts.append(len(video_clicks))
            judge_reads.append(
                JudgeClickCalibrationRead(
                    assignment_id=assignment.id,
                    display_name=assignment.display_name,
                    click_count=len(video_clicks),
                    matches=[
                        ClickMatchRead(
                            click_id=match.click.click_id,
                            timestamp_ms=match.click.timestamp_ms,
                            label=match.click.associated_label,
                            matched_event_label=match.matched_event_label,
                            boundary_error_ms=match.boundary_error_ms,
                        )
                        for match in matches
                    ],
                    model_event_count=agreement.model_event_count,
                    precision=agreement.precision if events else None,
                    recall=agreement.recall if ml_clicks else None,
                    mean_boundary_error_ms=agreement.mean_boundary_error_ms,
                )
            )

        panel_mean = (
            sum(panel_counts) / len(panel_counts) if panel_counts else None
        )
        video_reads.append(
            VideoClickCalibrationRead(
                entry_video_id=entry_video.id,
                original_filename=entry_video.video.original_filename,
                official_analysis_id=entry_video.official_analysis_id,
                model_event_count=len(events),
                judges=judge_reads,
                panel_click_count=sum(panel_counts),
                panel_mean_clicks=panel_mean,
            )
        )

    return JudgingEntryCalibrationRead(
        entry_id=entry.id,
        title=entry.title,
        click_mode=entry.click_mode,
        tolerance_ms=tolerance_ms,
        videos=video_reads,
        warnings=warnings,
    )
