"""Canonical, versioned dataset schema for the YoYoVision 1A annotation corpus.

Pydantic v2 models only -- no SQLAlchemy/FastAPI dependency, so this schema
can be reused by offline dataset tooling, training code (Prompt B/C), and
tests without pulling in the `api` package. Reuses `yoyovision_ml.domain`
enums (`EventFamily`, `Outcome`, `DifficultyBand`, `DeductionType`, `Source`)
directly so annotation vocabulary never drifts from the runtime pipeline's
vocabulary.

All timestamps are integer milliseconds (never float seconds), per product
principle #9 (preserve original video timestamps) and to keep exact
equality/ordering comparisons well-defined during validation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from yoyovision_ml.domain import DeductionType, DifficultyBand, EventFamily, Outcome, Source

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
#: Freestyle Evaluation categories are judge-entered on a 0-10 scale, matching
#: `scoring_engine.py`'s assumption ("each category assumed entered on a 0-10
#: scale by the judge/reviewer").
FreestyleScore = Annotated[float, Field(ge=0.0, le=10.0)]


class VisibilityState(StrEnum):
    """Frame-level visibility of an annotated subject (yo-yo, hand, body joint)."""

    VISIBLE = "visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    FULLY_OCCLUDED = "fully_occluded"
    OUTSIDE_FRAME = "outside_frame"
    UNLABELLED = "unlabelled"


class SplitName(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
class AnnotationProvenance(BaseModel):
    """Who/what produced an annotation, and whether it has been adjudicated.

    Required on every individual annotation item (not just at the record
    level) so a single record that blends human-drawn events with
    imported/model-assisted pre-labels still traces each item to its origin.
    """

    annotator_id: str
    source: Source = Source.HUMAN
    annotated_at: datetime
    tool: str = "manual"
    tool_version: str | None = None
    is_adjudicated: bool = False
    adjudicated_by: str | None = None
    adjudication_notes: str = ""

    @model_validator(mode="after")
    def _adjudication_requires_adjudicator(self) -> AnnotationProvenance:
        if self.is_adjudicated and not self.adjudicated_by:
            raise ValueError("is_adjudicated=True requires adjudicated_by to be set")
        return self


# --------------------------------------------------------------------------- #
# Video metadata
# --------------------------------------------------------------------------- #
class DatasetVideo(BaseModel):
    video_id: str
    player_id: str
    division: Literal["1A"] = "1A"
    relative_path: str
    checksum_sha256: str
    duration_ms: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_fps: float = Field(gt=0)
    is_variable_fps: bool = False
    #: Required (and validated against `duration_ms`/`source_fps`) when
    #: `is_variable_fps=True`, since a constant nominal fps cannot represent
    #: true per-frame timing in that case.
    frame_timestamps_ms: list[int] | None = None
    consent_reference: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _relative_path_is_safe(self) -> DatasetVideo:
        if self.relative_path.startswith(("/", "\\")) or ".." in self.relative_path.split("/"):
            raise ValueError(f"relative_path must be a safe relative path: {self.relative_path!r}")
        return self

    @model_validator(mode="after")
    def _variable_fps_requires_frame_timestamps(self) -> DatasetVideo:
        if self.is_variable_fps and not self.frame_timestamps_ms:
            raise ValueError("is_variable_fps=True requires non-empty frame_timestamps_ms")
        if self.frame_timestamps_ms:
            if self.frame_timestamps_ms != sorted(self.frame_timestamps_ms):
                raise ValueError("frame_timestamps_ms must be non-decreasing")
            if self.frame_timestamps_ms[-1] > self.duration_ms:
                raise ValueError("frame_timestamps_ms must not exceed duration_ms")
        return self


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
class NormalizedPoint(BaseModel):
    """Image-plane coordinates normalized to [0, 1] of frame width/height."""

    x: Fraction
    y: Fraction


class NormalizedBBox(BaseModel):
    x: Fraction
    y: Fraction
    width: Fraction
    height: Fraction

    @model_validator(mode="after")
    def _box_fits_in_frame(self) -> NormalizedBBox:
        if self.x + self.width > 1.0 + 1e-6 or self.y + self.height > 1.0 + 1e-6:
            raise ValueError("bbox extends beyond the normalized [0, 1] frame")
        return self


class AnnotatedKeypoint(BaseModel):
    name: str
    x: Fraction
    y: Fraction
    visibility: VisibilityState


# --------------------------------------------------------------------------- #
# Per-frame annotation streams
# --------------------------------------------------------------------------- #
class PoseLandmarkFrame(BaseModel):
    frame_ms: int = Field(ge=0)
    keypoints: list[AnnotatedKeypoint]


class HandLandmarkFrame(BaseModel):
    frame_ms: int = Field(ge=0)
    handedness: Literal["left", "right"]
    keypoints: list[AnnotatedKeypoint]


class YoyoFrameAnnotation(BaseModel):
    frame_ms: int = Field(ge=0)
    point: NormalizedPoint | None = None
    bbox: NormalizedBBox | None = None
    visibility: VisibilityState
    #: Set when this frame's position was produced/pre-filled by a model
    #: (`AnnotationProvenance.source == model`) rather than drawn by a human;
    #: `None` for pure human ground truth, which is not "confident", it's certain.
    confidence: Fraction | None = None

    @model_validator(mode="after")
    def _position_required_unless_unobservable(self) -> YoyoFrameAnnotation:
        unobservable = {VisibilityState.FULLY_OCCLUDED, VisibilityState.OUTSIDE_FRAME}
        if self.visibility not in unobservable and self.point is None and self.bbox is None:
            raise ValueError(
                f"frame_ms={self.frame_ms}: point or bbox is required when visibility="
                f"'{self.visibility}' (only fully_occluded/outside_frame may omit position)"
            )
        return self


class StringMaskFrame(BaseModel):
    """A frame-level string annotation.

    `mask_key` is a reference to a mask asset stored elsewhere (e.g. a PNG in
    the dataset's storage layout), never inline pixel data. When the string
    is not reliably observable, `observable=False` and `mask_key=None` --
    the schema deliberately cannot represent a "guessed" string mask.
    """

    frame_ms: int = Field(ge=0)
    observable: bool = True
    mask_key: str | None = None
    confidence: Fraction | None = None
    note: str = ""

    @model_validator(mode="after")
    def _mask_key_only_when_observable(self) -> StringMaskFrame:
        if not self.observable and self.mask_key is not None:
            raise ValueError("mask_key must be None when observable=False")
        if self.observable and self.mask_key is None:
            raise ValueError("mask_key is required when observable=True")
        return self


# --------------------------------------------------------------------------- #
# Event-level annotations
# --------------------------------------------------------------------------- #
class TrickEventAnnotation(BaseModel):
    event_id: str
    label: str
    family: EventFamily
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    outcome: Outcome
    difficulty_band: DifficultyBand = DifficultyBand.UNKNOWN
    confidence: Fraction = 1.0
    provenance: AnnotationProvenance
    notes: str = ""

    @model_validator(mode="after")
    def _end_after_start(self) -> TrickEventAnnotation:
        if self.end_ms <= self.start_ms:
            raise ValueError(
                f"event_id={self.event_id}: end_ms ({self.end_ms}) must be > "
                f"start_ms ({self.start_ms})"
            )
        return self


class DeductionAnnotation(BaseModel):
    deduction_id: str
    type: DeductionType
    timestamp_ms: int = Field(ge=0)
    quantity: int = Field(default=1, ge=1)
    confidence: Fraction = 1.0
    provenance: AnnotationProvenance
    notes: str = ""


class JudgeClickAnnotation(BaseModel):
    """A raw, low-effort timestamp click from a human judge watching the
    routine live or on review -- used later (Prompt D) to check how closely
    model-detected event boundaries track a judge's real-time perception,
    not treated as a fully-specified event itself."""

    click_id: str
    judge_id: str
    timestamp_ms: int = Field(ge=0)
    associated_label: str | None = None
    notes: str = ""


class FreestyleEvaluationAnnotation(BaseModel):
    judge_id: str
    execution: FreestyleScore | None = None
    control: FreestyleScore | None = None
    trick_diversity: FreestyleScore | None = None
    space_use_emphasis: FreestyleScore | None = None
    music_choreography: FreestyleScore | None = None
    music_construction: FreestyleScore | None = None
    body_control: FreestyleScore | None = None
    showmanship: FreestyleScore | None = None
    provenance: AnnotationProvenance
    notes: str = ""


# --------------------------------------------------------------------------- #
# Record: one annotation pass over one video
# --------------------------------------------------------------------------- #
class DatasetRecord(BaseModel):
    """One complete annotation pass (by one annotator, or the adjudicated
    merge of several) over a single `DatasetVideo`.

    Multiple `DatasetRecord`s may reference the same `video_id` -- one per
    independent annotator plus, once reconciled, one with
    `is_adjudicated=True` -- which is how "multiple annotators and
    adjudication" (requirement 6) and reviewer-agreement statistics
    (`dataset/stats.py`) are supported without a separate annotation-tool
    schema.
    """

    record_id: str
    video: DatasetVideo
    annotator_id: str
    is_adjudicated: bool = False
    schema_version: Literal["1.0.0"] = "1.0.0"
    ontology_version: str
    pose_landmarks: list[PoseLandmarkFrame] = Field(default_factory=list)
    hand_landmarks: list[HandLandmarkFrame] = Field(default_factory=list)
    yoyo_track: list[YoyoFrameAnnotation] = Field(default_factory=list)
    string_masks: list[StringMaskFrame] = Field(default_factory=list)
    trick_events: list[TrickEventAnnotation] = Field(default_factory=list)
    deductions: list[DeductionAnnotation] = Field(default_factory=list)
    judge_clicks: list[JudgeClickAnnotation] = Field(default_factory=list)
    freestyle_evaluations: list[FreestyleEvaluationAnnotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_duplicate_item_ids(self) -> DatasetRecord:
        for id_field, items in (
            ("event_id", self.trick_events),
            ("deduction_id", self.deductions),
            ("click_id", self.judge_clicks),
        ):
            ids = [getattr(item, id_field) for item in items]
            duplicates = {i for i in ids if ids.count(i) > 1}
            if duplicates:
                raise ValueError(
                    f"record_id={self.record_id}: duplicate {id_field} values: {sorted(duplicates)}"
                )
        return self


class DatasetManifest(BaseModel):
    """Top-level index for a versioned dataset directory."""

    dataset_version: str
    ontology_version: str
    created_at: datetime
    video_ids: list[str]
    record_paths: list[str]
    split_seed: int | None = None
    splits: dict[str, SplitName] | None = None  # video_id -> split
    notes: str = ""
