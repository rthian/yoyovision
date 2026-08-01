"""Pydantic v2 request/response schemas for the public API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from yoyovision_ml.pipeline_config import PipelineAdapterConfig
from yoyovision_api.judging_enums import (
    AggregationMode,
    AiMixProfile,
    JudgeAssignmentStatus,
    JudgingEntryMode,
    JudgingEntryStatus,
)
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


class VideoAssetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    owner_id: str
    original_filename: str
    mime_type: str
    duration_ms: int | None
    width: int | None
    height: int | None
    fps: float | None
    file_size: int
    status: VideoStatus
    created_at: datetime
    deleted_at: datetime | None


class AnalysisJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    video_id: str
    status: JobStatus
    progress: float
    current_stage: PipelineStage | None
    error_code: str | None
    error_message: str | None
    pipeline_version: str
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    #: Prompt F (production inference) fields. `model_versions` only ever
    #: holds `name@version` strings (never a local filesystem path -- see
    #: `ModelRegistry.describe`).
    model_versions: dict[str, str] | None = None
    device: str | None = None
    runtime_versions: dict[str, str] | None = None
    stage_durations_ms: dict[str, float] | None = None
    #: Shadow jobs run the full pipeline and persist real results, but are
    #: never meant to be treated as a video's official/canonical score.
    is_shadow: bool = False
    cancel_requested: bool = False
    retry_count: int = 0
    #: Optional judged routine window within the uploaded clip (measure start /
    #: music stop). Null means "use full video duration".
    routine_start_ms: int | None = None
    routine_end_ms: int | None = None
    review_state: AnalysisReviewState = AnalysisReviewState.DRAFT
    submitted_at: datetime | None = None
    ruleset_version: str = "1a-draft-0.1"
    pipeline_adapter_config: PipelineAdapterConfig | None = None


class RoutineWindowUpdate(BaseModel):
    routine_start_ms: int | None = Field(default=None, ge=0)
    routine_end_ms: int | None = Field(default=None, ge=0)


class RulesetVersionUpdate(BaseModel):
    ruleset_version: str = Field(min_length=1, max_length=32)


class PipelineAdapterConfigUpdate(BaseModel):
    pipeline_adapter_config: PipelineAdapterConfig | None = None


class AnalysisJobCreate(BaseModel):
    pipeline_adapter_config: PipelineAdapterConfig | None = None


class AnalysisEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    analysis_id: str
    label: str
    family: EventFamily
    start_ms: int
    end_ms: int
    confidence: float
    outcome: Outcome
    difficulty_band: DifficultyBand
    source: Source
    review_status: ReviewStatus
    model_name: str | None
    model_version: str | None
    evidence_json: dict[str, object]
    created_at: datetime
    updated_at: datetime


class AnalysisEventCreate(BaseModel):
    """Used when a human adds a new event manually during review."""

    label: str = Field(min_length=1, max_length=128)
    family: EventFamily
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    outcome: Outcome
    difficulty_band: DifficultyBand = DifficultyBand.UNKNOWN
    notes: str | None = None


class AnalysisEventUpdate(BaseModel):
    """Partial update -- any provided field is edited; source becomes `human`
    if not already, and review_status moves to `edited` unless explicitly set."""

    label: str | None = Field(default=None, min_length=1, max_length=128)
    family: EventFamily | None = None
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    outcome: Outcome | None = None
    difficulty_band: DifficultyBand | None = None
    review_status: ReviewStatus | None = None


class MajorDeductionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    analysis_id: str
    type: DeductionType
    timestamp_ms: int
    quantity: int
    points: float
    confidence: float
    source: Source
    review_status: ReviewStatus


class MajorDeductionCreate(BaseModel):
    type: DeductionType
    timestamp_ms: int = Field(ge=0)
    quantity: int = Field(default=1, ge=1)
    points: float = Field(ge=0.0)


class MajorDeductionUpdate(BaseModel):
    type: DeductionType | None = None
    timestamp_ms: int | None = Field(default=None, ge=0)
    quantity: int | None = Field(default=None, ge=1)
    points: float | None = Field(default=None, ge=0.0)
    review_status: ReviewStatus | None = None


class FreestyleEvaluationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    execution: float | None
    control: float | None
    trick_diversity: float | None
    space_use_emphasis: float | None
    music_choreography: float | None
    music_construction: float | None
    body_control: float | None
    showmanship: float | None
    source: Source
    notes: str


class FreestyleEvaluationUpsert(BaseModel):
    """Manual entry per MVP scope ("Freestyle Evaluation placeholders and
    manual values"); every field 0-10, all optional (partial entry allowed)."""

    execution: float | None = Field(default=None, ge=0.0, le=10.0)
    control: float | None = Field(default=None, ge=0.0, le=10.0)
    trick_diversity: float | None = Field(default=None, ge=0.0, le=10.0)
    space_use_emphasis: float | None = Field(default=None, ge=0.0, le=10.0)
    music_choreography: float | None = Field(default=None, ge=0.0, le=10.0)
    music_construction: float | None = Field(default=None, ge=0.0, le=10.0)
    body_control: float | None = Field(default=None, ge=0.0, le=10.0)
    showmanship: float | None = Field(default=None, ge=0.0, le=10.0)
    notes: str = Field(default="", max_length=4096)


class ScoreBreakdownRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    technical_raw: float
    technical_scaled: float
    freestyle_evaluation_raw: float
    freestyle_evaluation_scaled: float
    major_deductions: float
    final_score: float
    confidence: float
    ruleset_version: str
    warnings: list[str]


class TechnicalLineItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str | None
    start_ms: int
    label: str
    family: EventFamily
    base_points: float
    multiplier: float
    points: float
    reason: str


class ScoreLineItemsRead(BaseModel):
    technical_raw: float
    technical_line_items: list[TechnicalLineItemRead]


class ScorePreviewRead(BaseModel):
    up_to_ms: int
    completed_event_count: int
    active_event_id: str | None
    technical_raw: float
    technical_scaled: float
    freestyle_evaluation_raw: float
    freestyle_evaluation_scaled: float
    major_deductions: float
    final_score: float
    confidence: float
    ruleset_version: str
    warnings: list[str]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: str
    password: str


class CorpusExportRead(BaseModel):
    record_id: str
    record_path: str
    corpus_root: str
    video_path: str


# --- Multi-judge entries (Phase B) ---


class JudgingEntryCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    mode: JudgingEntryMode
    ruleset_version: str | None = None
    ai_mix_profile: AiMixProfile = AiMixProfile.COMPARE_ONLY
    aggregation_mode: AggregationMode = AggregationMode.AUTO
    due_at: datetime | None = None
    video_ids: list[str] = Field(min_length=1)


class JudgingEntryUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    mode: JudgingEntryMode | None = None
    status: JudgingEntryStatus | None = None
    ruleset_version: str | None = None
    ai_mix_profile: AiMixProfile | None = None
    aggregation_mode: AggregationMode | None = None
    due_at: datetime | None = None
    clear_due_at: bool = False


class JudgingEntryVideoAttach(BaseModel):
    video_ids: list[str] = Field(min_length=1)


class JudgingEntryVideoAnalysisLink(BaseModel):
    official_analysis_id: str | None = None
    shadow_analysis_id: str | None = None


class JudgeAssignmentCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=128)
    include_in_results: bool = True
    is_shadow: bool = False


class JudgeInviteRead(BaseModel):
    assignment_id: str
    display_name: str
    token_prefix: str
    invite_url: str
    share_message: str
    token_expires_at: datetime
    include_in_results: bool
    is_shadow: bool
    status: JudgeAssignmentStatus


class JudgeAssignmentSummary(BaseModel):
    id: str
    display_name: str
    token_prefix: str
    token_expires_at: datetime
    include_in_results: bool
    is_shadow: bool
    revoked_at: datetime | None
    status: JudgeAssignmentStatus


class JudgingEntryVideoRead(BaseModel):
    id: str
    video_id: str
    sort_order: int
    original_filename: str
    official_analysis_id: str | None
    shadow_analysis_id: str | None


class JudgingEntryRead(BaseModel):
    id: str
    title: str
    mode: JudgingEntryMode
    status: JudgingEntryStatus
    ruleset_version: str
    ai_mix_profile: AiMixProfile
    aggregation_mode: AggregationMode
    due_at: datetime | None
    created_at: datetime
    updated_at: datetime
    videos: list[JudgingEntryVideoRead]
    judges: list[JudgeAssignmentSummary]
