"""SQLAlchemy 2.0 ORM models mirroring the domain models in the spec.

Enums reuse `yoyovision_ml.domain` enums directly so the API layer, workers
layer, and ml layer never drift out of sync on vocabulary.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from yoyovision_ml.domain import (
    AnalysisReviewState,
    DeductionType,
    DifficultyBand,
    EventFamily,
    JobStatus,
    Outcome,
    PipelineStage,
    ReviewStatus,
    Source,
    VideoStatus,
)

from yoyovision_api.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _str_enum(enum_cls: type, length: int) -> SAEnum:
    """`Enum` column that persists/reads each member's `.value` (e.g.
    `"completed"`) instead of SQLAlchemy's default `.name` (e.g. `"COMPLETED"`).

    Required because `workers/src/yoyovision_workers/schema.py` writes these
    same columns through plain `String` Core columns using `JobStatus.X.value`
    (see `pipeline_runner.py`) -- without `values_callable` here, the ORM's
    default `.name`-based round-trip does not match those rows and raises
    `LookupError` the next time the API reads them back.
    """
    return SAEnum(
        enum_cls,
        native_enum=False,
        length=length,
        values_callable=lambda obj: [e.value for e in obj],
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class VideoAssetORM(Base):
    __tablename__ = "video_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), index=True, nullable=False
    )
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[VideoStatus] = mapped_column(
        _str_enum(VideoStatus, 32),
        nullable=False,
        default=VideoStatus.UPLOADED,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    jobs: Mapped[list[AnalysisJobORM]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )


class AnalysisJobORM(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    video_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("video_assets.id"), index=True, nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        _str_enum(JobStatus, 32), nullable=False, default=JobStatus.PENDING
    )
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    current_stage: Mapped[PipelineStage | None] = mapped_column(
        _str_enum(PipelineStage, 32), nullable=True
    )
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: Prompt F (production inference) columns. All nullable/defaulted so
    #: pre-Prompt-F jobs (and the mock-only test fixtures) read back fine.
    #: `model_versions`/`runtime_versions`/`stage_durations_ms` only ever
    #: hold `name@version` strings, interpreter/runtime version strings, and
    #: stage-name -> millisecond mappings -- never a local filesystem path
    #: (see `inference.model_registry.ModelRegistry.describe`).
    model_versions: Mapped[dict[str, str] | None] = mapped_column(JSON, nullable=True)
    device: Mapped[str | None] = mapped_column(String(32), nullable=True)
    runtime_versions: Mapped[dict[str, str] | None] = mapped_column(JSON, nullable=True)
    stage_durations_ms: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    #: Shadow-mode jobs run the full pipeline and persist real results, but
    #: are never treated as "the" official score for a video (Prompt F:
    #: "shadow mode that computes model results without exposing them as
    #: final scores") -- see `services/job_service.py` and `routers/videos.py`.
    is_shadow: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    #: Set by `POST /analyses/{id}/cancel`; polled cooperatively by the
    #: worker between pipeline stages via `CancellationToken.cancel_check`.
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    #: Incremented by the worker each time a `TransientPipelineError` causes
    #: a Celery retry -- never incremented for deterministic failures.
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Judged routine window inside the uploaded clip. When set, scoring and the
    #: review UI treat only this span as the competitive routine.
    routine_start_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    routine_end_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    review_state: Mapped[AnalysisReviewState] = mapped_column(
        _str_enum(AnalysisReviewState, 16),
        nullable=False,
        default=AnalysisReviewState.DRAFT,
    )
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    submitted_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True
    )
    #: Versioned scoring config applied to this analysis. Defaults from API
    #: settings at job creation; judges may switch rulesets during review.
    ruleset_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="1a-draft-0.1"
    )

    video: Mapped[VideoAssetORM] = relationship(back_populates="jobs")
    events: Mapped[list[AnalysisEventORM]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    deductions: Mapped[list[MajorDeductionORM]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    freestyle_evaluation: Mapped[FreestyleEvaluationORM | None] = relationship(
        back_populates="analysis", cascade="all, delete-orphan", uselist=False
    )
    score_breakdown: Mapped[ScoreBreakdownORM | None] = relationship(
        back_populates="analysis", cascade="all, delete-orphan", uselist=False
    )


class AnalysisEventORM(Base):
    __tablename__ = "analysis_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_jobs.id"), index=True, nullable=False
    )
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    family: Mapped[EventFamily] = mapped_column(_str_enum(EventFamily, 48), nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    outcome: Mapped[Outcome] = mapped_column(_str_enum(Outcome, 16), nullable=False)
    difficulty_band: Mapped[DifficultyBand] = mapped_column(
        _str_enum(DifficultyBand, 16), nullable=False
    )
    source: Mapped[Source] = mapped_column(_str_enum(Source, 16), nullable=False)
    review_status: Mapped[ReviewStatus] = mapped_column(
        _str_enum(ReviewStatus, 16),
        nullable=False,
        default=ReviewStatus.PENDING,
    )
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    analysis: Mapped[AnalysisJobORM] = relationship(back_populates="events")


class MajorDeductionORM(Base):
    __tablename__ = "major_deductions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_jobs.id"), index=True, nullable=False
    )
    type: Mapped[DeductionType] = mapped_column(_str_enum(DeductionType, 32), nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    points: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[Source] = mapped_column(_str_enum(Source, 16), nullable=False)
    review_status: Mapped[ReviewStatus] = mapped_column(
        _str_enum(ReviewStatus, 16),
        nullable=False,
        default=ReviewStatus.PENDING,
    )

    analysis: Mapped[AnalysisJobORM] = relationship(back_populates="deductions")


class FreestyleEvaluationORM(Base):
    __tablename__ = "freestyle_evaluations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_jobs.id"), unique=True, nullable=False
    )
    execution: Mapped[float | None] = mapped_column(Float, nullable=True)
    control: Mapped[float | None] = mapped_column(Float, nullable=True)
    trick_diversity: Mapped[float | None] = mapped_column(Float, nullable=True)
    space_use_emphasis: Mapped[float | None] = mapped_column(Float, nullable=True)
    music_choreography: Mapped[float | None] = mapped_column(Float, nullable=True)
    music_construction: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_control: Mapped[float | None] = mapped_column(Float, nullable=True)
    showmanship: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[Source] = mapped_column(
        _str_enum(Source, 16), nullable=False, default=Source.HUMAN
    )
    notes: Mapped[str] = mapped_column(String(4096), nullable=False, default="")

    analysis: Mapped[AnalysisJobORM] = relationship(back_populates="freestyle_evaluation")


class ScoreBreakdownORM(Base):
    __tablename__ = "score_breakdowns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_jobs.id"), unique=True, nullable=False
    )
    technical_raw: Mapped[float] = mapped_column(Float, nullable=False)
    technical_scaled: Mapped[float] = mapped_column(Float, nullable=False)
    freestyle_evaluation_raw: Mapped[float] = mapped_column(Float, nullable=False)
    freestyle_evaluation_scaled: Mapped[float] = mapped_column(Float, nullable=False)
    major_deductions: Mapped[float] = mapped_column(Float, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    ruleset_version: Mapped[str] = mapped_column(String(32), nullable=False)
    warnings: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    analysis: Mapped[AnalysisJobORM] = relationship(back_populates="score_breakdown")
