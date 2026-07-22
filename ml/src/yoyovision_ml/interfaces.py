"""Protocol interfaces for every replaceable ML/domain component.

Per product principle #5 ("Model adapters must be replaceable"), no calling
code should ever import a concrete adapter class directly for its behaviour --
only these Protocols, resolved through `adapters_registry.py`. This keeps the
product decoupled from any single detector, tracker, or model vendor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from yoyovision_ml.domain import (
    AnalysisEventPrediction,
    DeductionPrediction,
    Detection,
    FeatureSet,
    FreestyleEvaluation,
    HandSequence,
    PoseSequence,
    ScoreBreakdown,
    Track,
)
from yoyovision_ml.ruleset import Ruleset


class FrameRef:
    """Lightweight reference to a decoded frame, kept out of `domain.py`.

    Deliberately not a dataclass with numpy inline to avoid forcing every
    domain.py consumer to depend on numpy; adapters that need raw pixels
    import numpy themselves and treat `.array` as an `np.ndarray`.
    """

    __slots__ = ("frame_ms", "array")

    def __init__(self, frame_ms: int, array: object) -> None:
        self.frame_ms = frame_ms
        self.array = array


@runtime_checkable
class PoseEstimator(Protocol):
    """Extracts full-body pose landmarks across a video."""

    model_name: str
    model_version: str

    def predict(self, video_path: Path) -> PoseSequence: ...


@runtime_checkable
class HandEstimator(Protocol):
    """Extracts hand landmarks (both hands) across a video."""

    model_name: str
    model_version: str

    def predict(self, video_path: Path) -> HandSequence: ...


@runtime_checkable
class YoyoDetector(Protocol):
    """Per-frame yo-yo object detection."""

    model_name: str
    model_version: str

    def predict(self, frame_batch: list[FrameRef]) -> list[Detection]: ...


@runtime_checkable
class ObjectTracker(Protocol):
    """Associates per-frame detections into persistent tracks over time."""

    model_name: str
    model_version: str

    def update(self, detections: list[Detection], timestamp_ms: int) -> list[Track]: ...

    def reset(self) -> None: ...


@runtime_checkable
class StringAnalyzer(Protocol):
    """Estimates string/slack geometry from tracked positions (yo-yo, hands)."""

    model_name: str
    model_version: str

    def analyze(self, yoyo_track: list[Track], hand_sequence: HandSequence) -> FeatureSet: ...


@runtime_checkable
class RgbEncoder(Protocol):
    """Extracts appearance-based (RGB pixel) signal per frame -- Prompt E's
    complement to the kinematic (pose/hand/track-geometry) feature stack,
    e.g. a visual embedding summarizing scene/appearance detail that pure
    keypoint/bbox tracking cannot capture (lighting, background, prop
    color/shape detail)."""

    model_name: str
    model_version: str

    def encode(self, frame_batch: list[FrameRef]) -> FeatureSet: ...


@runtime_checkable
class StringSegmenter(Protocol):
    """Pixel-based string/slack segmentation -- a richer, Prompt E signal
    than `StringAnalyzer`'s hand<->yoyo geometric-distance proxy. Intended
    to eventually estimate string visibility/angle/slack from a real
    segmentation mask rather than inferring it purely from tracked
    bounding-box positions; `StringAnalyzer` stays in the pipeline
    unchanged as the always-available geometric fallback."""

    model_name: str
    model_version: str

    def segment(self, frame_batch: list[FrameRef], yoyo_track: list[Track]) -> FeatureSet: ...


@runtime_checkable
class AudioAnalyzer(Protocol):
    """Extracts audio-track features (tempo/onset/energy) for Prompt E's
    multimodal fusion. Distinct from `scoring.fe_estimators`'s manual
    Freestyle Evaluation heuristics, which have no audio input today (see
    `_NO_AUDIO_WARNING`) -- this is a temporal-event-detection input, not a
    Freestyle Evaluation scoring input."""

    model_name: str
    model_version: str

    def analyze(self, video_path: Path, duration_ms: int) -> FeatureSet: ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Combines pose/hand/yoyo-track/string signals into a unified per-frame feature set."""

    def extract(
        self,
        pose_sequence: PoseSequence,
        hand_sequence: HandSequence,
        yoyo_tracks: list[Track],
        string_features: FeatureSet,
    ) -> FeatureSet: ...


@runtime_checkable
class TemporalEventDetector(Protocol):
    """Segments the feature timeline into atomic trick events and equipment deductions."""

    model_name: str
    model_version: str

    def predict(
        self, features: FeatureSet
    ) -> tuple[list[AnalysisEventPrediction], list[DeductionPrediction]]: ...


@runtime_checkable
class ScoringEngine(Protocol):
    """Deterministic, rules-based scoring -- never an opaque score predictor."""

    def calculate(
        self,
        events: list[AnalysisEventPrediction],
        deductions: list[DeductionPrediction],
        freestyle_evaluation: FreestyleEvaluation | None,
        ruleset: Ruleset,
    ) -> ScoreBreakdown: ...


@runtime_checkable
class StoragePort(Protocol):
    """Storage abstraction so local-filesystem and S3-compatible backends are interchangeable."""

    def put(self, storage_key: str, data: bytes, content_type: str) -> None: ...

    def get(self, storage_key: str) -> bytes: ...

    def delete(self, storage_key: str) -> None: ...

    def exists(self, storage_key: str) -> bool: ...

    def signed_url(self, storage_key: str, expires_seconds: int) -> str: ...
