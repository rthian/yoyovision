"""Converts persisted ORM rows into the framework-agnostic `yoyovision_ml`
domain dataclasses, used by the export endpoints (`yoyovision_ml.exports`
operates purely on those dataclasses, with no ORM/SQLAlchemy dependency)."""

from __future__ import annotations

from yoyovision_ml.domain import AnalysisEvent, MajorDeduction, ScoreBreakdown, VideoAsset

from yoyovision_api.db_models import (
    AnalysisEventORM,
    MajorDeductionORM,
    ScoreBreakdownORM,
    VideoAssetORM,
)


def video_to_domain(video: VideoAssetORM) -> VideoAsset:
    return VideoAsset(
        id=video.id,
        owner_id=video.owner_id,
        original_filename=video.original_filename,
        storage_key=video.storage_key,
        mime_type=video.mime_type,
        duration_ms=video.duration_ms,
        width=video.width,
        height=video.height,
        fps=video.fps,
        file_size=video.file_size,
        status=video.status,
        created_at=video.created_at,
        deleted_at=video.deleted_at,
    )


def event_to_domain(event: AnalysisEventORM) -> AnalysisEvent:
    return AnalysisEvent(
        id=event.id,
        analysis_id=event.analysis_id,
        label=event.label,
        family=event.family,
        start_ms=event.start_ms,
        end_ms=event.end_ms,
        confidence=event.confidence,
        outcome=event.outcome,
        difficulty_band=event.difficulty_band,
        source=event.source,
        review_status=event.review_status,
        model_name=event.model_name,
        model_version=event.model_version,
        evidence_json=event.evidence_json,
        created_at=event.created_at,
        updated_at=event.updated_at,
    )


def deduction_to_domain(deduction: MajorDeductionORM) -> MajorDeduction:
    return MajorDeduction(
        id=deduction.id,
        analysis_id=deduction.analysis_id,
        type=deduction.type,
        timestamp_ms=deduction.timestamp_ms,
        quantity=deduction.quantity,
        points=deduction.points,
        confidence=deduction.confidence,
        source=deduction.source,
        review_status=deduction.review_status,
    )


def score_to_domain(score: ScoreBreakdownORM) -> ScoreBreakdown:
    return ScoreBreakdown(
        technical_raw=score.technical_raw,
        technical_scaled=score.technical_scaled,
        freestyle_evaluation_raw=score.freestyle_evaluation_raw,
        freestyle_evaluation_scaled=score.freestyle_evaluation_scaled,
        major_deductions=score.major_deductions,
        final_score=score.final_score,
        confidence=score.confidence,
        ruleset_version=score.ruleset_version,
        warnings=list(score.warnings),
    )
