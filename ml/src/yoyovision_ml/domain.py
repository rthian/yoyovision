"""Framework-agnostic domain vocabulary shared by the api, workers, and ml packages.

These are plain dataclasses/enums with no dependency on SQLAlchemy, Pydantic, or any
web framework. The `api` package maps these to ORM models and Pydantic schemas; the
`workers` package produces/consumes them while running the analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class VideoStatus(StrEnum):
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    READY = "ready"
    REJECTED = "rejected"
    DELETED = "deleted"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStage(StrEnum):
    """Stages surfaced as `AnalysisJob.current_stage` for observability."""

    QUEUED = "queued"
    MEDIA_VALIDATION = "media_validation"
    PREPROCESSING = "preprocessing"
    POSE_EXTRACTION = "pose_extraction"
    HAND_EXTRACTION = "hand_extraction"
    YOYO_DETECTION = "yoyo_detection"
    TRACKING = "tracking"
    STRING_ANALYSIS = "string_analysis"
    FEATURE_EXTRACTION = "feature_extraction"
    TEMPORAL_EVENT_DETECTION = "temporal_event_detection"
    SCORING = "scoring"
    DONE = "done"


class EventFamily(StrEnum):
    """The initial supported 1A atomic trick-element families."""

    MOUNT = "mount"
    HOP = "hop"
    LACERATION = "laceration"
    WHIP_CATCH = "whip_catch"
    SLACK = "slack"
    SUICIDE = "suicide"
    REJECTION = "rejection"
    ROLL = "roll"
    UNDERPASS = "underpass"
    OVERPASS = "overpass"
    BIND = "bind"
    RETURN = "return"
    REGENERATION = "regeneration"
    HORIZONTAL = "horizontal"
    FINGERSPIN = "fingerspin"
    BODY_TRICK = "body_trick"
    CONTROL_MISS = "control_miss"
    LANDING_MISS = "landing_miss"
    CATCH_MISS = "catch_miss"
    YOYO_STOP = "yoyo_stop"
    YOYO_CHANGE = "yoyo_change"
    YOYO_DETACH = "yoyo_detach"
    UNKNOWN_TECHNICAL_ELEMENT = "unknown_technical_element"


#: Families that represent a positive/successful technical element attempt,
#: as opposed to a miss/mistake/equipment event. Used by the scoring engine
#: to decide whether an event can earn positive credit.
POSITIVE_EVENT_FAMILIES: frozenset[EventFamily] = frozenset(
    {
        EventFamily.MOUNT,
        EventFamily.HOP,
        EventFamily.LACERATION,
        EventFamily.WHIP_CATCH,
        EventFamily.SLACK,
        EventFamily.SUICIDE,
        EventFamily.REJECTION,
        EventFamily.ROLL,
        EventFamily.UNDERPASS,
        EventFamily.OVERPASS,
        EventFamily.BIND,
        EventFamily.RETURN,
        EventFamily.REGENERATION,
        EventFamily.HORIZONTAL,
        EventFamily.FINGERSPIN,
        EventFamily.BODY_TRICK,
    }
)

#: Families that represent a mistake/miss (never earn positive credit).
MISTAKE_EVENT_FAMILIES: frozenset[EventFamily] = frozenset(
    {
        EventFamily.CONTROL_MISS,
        EventFamily.LANDING_MISS,
        EventFamily.CATCH_MISS,
    }
)

#: Families that represent an equipment/major-deduction-triggering event.
EQUIPMENT_EVENT_FAMILIES: frozenset[EventFamily] = frozenset(
    {
        EventFamily.YOYO_STOP,
        EventFamily.YOYO_CHANGE,
        EventFamily.YOYO_DETACH,
    }
)


class Outcome(StrEnum):
    SUCCESS = "success"
    MISS = "miss"
    UNCERTAIN = "uncertain"


class DifficultyBand(StrEnum):
    """A model- or human-assigned difficulty estimate.

    This is NEVER an official published trick value from IYYF/WYYC or any
    other body -- see docs/ruleset.md. It is only used as an internal,
    versioned scoring input.
    """

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    UNKNOWN = "unknown"


class Source(StrEnum):
    MODEL = "model"
    HUMAN = "human"
    IMPORTED = "imported"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EDITED = "edited"


class DeductionType(StrEnum):
    YOYO_STOP = "yoyo_stop"
    YOYO_CHANGE = "yoyo_change"
    YOYO_DETACH = "yoyo_detach"
    #: Prompt D: a possible dangerous-play incident. Deliberately never
    #: auto-applied to a score -- see `ruleset.DeductionRule.requires_manual_confirmation`
    #: and `scoring_engine.deduction_is_scorable` -- it only ever creates a
    #: human review flag ("Dangerous-play detection must never automatically
    #: disqualify a player. It must create a review flag.").
    DANGEROUS_PLAY_REVIEW = "dangerous_play_review"
    OTHER = "other"


class VisibilityState(StrEnum):
    """Frame-level visibility of a tracked subject (yo-yo, hand, body joint).

    Mirrors `yoyovision_ml.dataset.schema.VisibilityState` (Prompt A's
    annotation-format enum) intentionally -- see docs/data_model.md. Kept as
    a separate definition here so this dependency-free `domain.py` never
    imports the Pydantic dataset schema.
    """

    VISIBLE = "visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    FULLY_OCCLUDED = "fully_occluded"
    OUTSIDE_FRAME = "outside_frame"
    UNLABELLED = "unlabelled"


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(slots=True, frozen=True)
class EvidenceRef:
    """A pointer back to the raw signal that justifies a detected event.

    Every detected event must be traceable to evidence per product principle #2.
    """

    frame_ms: int
    bbox: BoundingBox | None = None
    keypoint_refs: tuple[str, ...] = ()
    note: str = ""


# --------------------------------------------------------------------------- #
# Core persisted domain models (mirrors DB rows, framework-agnostic)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class VideoAsset:
    id: str
    owner_id: str
    original_filename: str
    storage_key: str
    mime_type: str
    duration_ms: int | None
    width: int | None
    height: int | None
    fps: float | None
    file_size: int
    status: VideoStatus
    created_at: datetime
    deleted_at: datetime | None = None


@dataclass(slots=True)
class AnalysisJob:
    id: str
    video_id: str
    status: JobStatus
    progress: float
    current_stage: PipelineStage | None
    error_code: str | None
    error_message: str | None
    pipeline_version: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(slots=True)
class AnalysisEvent:
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
    evidence_json: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class MajorDeduction:
    id: str
    analysis_id: str
    type: DeductionType
    timestamp_ms: int
    quantity: int
    points: float
    confidence: float
    source: Source
    review_status: ReviewStatus


@dataclass(slots=True)
class FreestyleEvaluation:
    execution: float | None
    control: float | None
    trick_diversity: float | None
    space_use_emphasis: float | None
    music_choreography: float | None
    music_construction: float | None
    body_control: float | None
    showmanship: float | None
    source: Source
    notes: str = ""


@dataclass(slots=True)
class ScoreBreakdown:
    technical_raw: float
    technical_scaled: float
    freestyle_evaluation_raw: float
    freestyle_evaluation_scaled: float
    major_deductions: float
    final_score: float
    confidence: float
    ruleset_version: str
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pipeline intermediate types (pose/hands/detections/tracks/features)
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    z: float | None
    visibility: float


@dataclass(slots=True, frozen=True)
class PoseFrame:
    frame_ms: int
    keypoints: tuple[Keypoint, ...]
    confidence: float


@dataclass(slots=True, frozen=True)
class PoseSequence:
    frames: tuple[PoseFrame, ...]
    model_name: str
    model_version: str
    fps: float


@dataclass(slots=True, frozen=True)
class HandFrame:
    frame_ms: int
    handedness: str  # "left" | "right"
    keypoints: tuple[Keypoint, ...]
    confidence: float


@dataclass(slots=True, frozen=True)
class HandSequence:
    frames: tuple[HandFrame, ...]
    model_name: str
    model_version: str
    fps: float


@dataclass(slots=True, frozen=True)
class Detection:
    frame_ms: int
    bbox: BoundingBox
    confidence: float
    class_label: str
    model_name: str
    model_version: str


@dataclass(slots=True, frozen=True)
class Track:
    track_id: str
    frame_ms: int
    bbox: BoundingBox
    confidence: float
    class_label: str
    #: Defaults preserve every pre-existing `Track(...)` call site. Real
    #: trackers (e.g. the Kalman baseline) set these explicitly per frame.
    visibility: VisibilityState = VisibilityState.VISIBLE
    interpolated: bool = False


@dataclass(slots=True, frozen=True)
class FeatureFrame:
    frame_ms: int
    values: dict[str, float]


@dataclass(slots=True, frozen=True)
class FeatureSet:
    frames: tuple[FeatureFrame, ...]
    feature_names: tuple[str, ...]
    fps: float


@dataclass(slots=True, frozen=True)
class AnalysisEventPrediction:
    """Raw model output before persistence -- becomes an `AnalysisEvent` row."""

    label: str
    family: EventFamily
    start_ms: int
    end_ms: int
    confidence: float
    outcome: Outcome
    difficulty_band: DifficultyBand
    model_name: str
    model_version: str
    evidence: tuple[EvidenceRef, ...] = ()


@dataclass(slots=True, frozen=True)
class DeductionPrediction:
    type: DeductionType
    timestamp_ms: int
    quantity: int
    confidence: float
    model_name: str
    model_version: str
