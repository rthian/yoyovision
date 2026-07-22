"""Deterministic mock inference adapters.

Per product principles #6 and #7: these adapters NEVER claim to be trained
models. Every instance sets `model_name` prefixed with `mock-` and a
`model_version` that makes clear no real weights are involved. Output is
generated from a stable hash of the input so results are fully reproducible
(same input -> same output every run), which is what "deterministic mock"
means here -- it is not a trained model, it does not learn, and its numbers
must never be presented as measured accuracy.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from yoyovision_ml.adapters_registry import (
    register_hand_estimator,
    register_pose_estimator,
    register_temporal_event_detector,
    register_tracker,
    register_yoyo_detector,
)
from yoyovision_ml.domain import (
    AnalysisEventPrediction,
    BoundingBox,
    DeductionPrediction,
    DeductionType,
    Detection,
    DifficultyBand,
    EventFamily,
    EvidenceRef,
    FeatureSet,
    HandFrame,
    HandSequence,
    Keypoint,
    Outcome,
    PoseFrame,
    PoseSequence,
    Track,
    VisibilityState,
)
from yoyovision_ml.interfaces import FrameRef
from yoyovision_ml.perception.landmarks import HAND_LANDMARK_NAMES, POSE_LANDMARK_NAMES

_MOCK_FPS_DEFAULT = 30.0
_MOCK_DURATION_MS_DEFAULT = 20_000

_POSE_LANDMARK_NAMES = POSE_LANDMARK_NAMES
_HAND_LANDMARK_NAMES = HAND_LANDMARK_NAMES

_EVENT_FAMILY_CYCLE: tuple[EventFamily, ...] = (
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
    EventFamily.CONTROL_MISS,
    EventFamily.LANDING_MISS,
    EventFamily.CATCH_MISS,
    EventFamily.UNKNOWN_TECHNICAL_ELEMENT,
)

_OUTCOME_CYCLE: tuple[Outcome, ...] = (
    Outcome.SUCCESS,
    Outcome.SUCCESS,
    Outcome.MISS,
    Outcome.UNCERTAIN,
)
_BAND_CYCLE: tuple[DifficultyBand, ...] = (
    DifficultyBand.BASIC,
    DifficultyBand.INTERMEDIATE,
    DifficultyBand.ADVANCED,
)


def _stable_seed(*parts: str) -> int:
    """Deterministic 64-bit seed independent of PYTHONHASHSEED randomization."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int(struct.unpack(">Q", digest[:8])[0])


def _deterministic_unit_floats(seed: int, count: int) -> list[float]:
    """A tiny xorshift-style PRNG so we avoid a `random.Random` dependency
    surprise across Python versions while staying deterministic."""
    values: list[float] = []
    state = seed or 1
    for _ in range(count):
        state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 7
        state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
        values.append((state % 10_000) / 10_000.0)
    return values


def _mock_video_timeline(
    video_path: Path,
    *,
    duration_ms: int | None = None,
    fps: float | None = None,
) -> tuple[float, int]:
    """Resolve fps and duration for mock pose/hand timelines.

    Mock adapters must span the uploaded clip, not a fixed 20s stub, so
    temporal event detection and scoring can cover the full routine.
    """
    if duration_ms is not None and duration_ms > 0 and fps is not None and fps > 0:
        return fps, duration_ms

    try:
        from yoyovision_ml.media_validation import probe_video_metadata

        metadata = probe_video_metadata(video_path)
        resolved_fps = metadata.fps if metadata.fps > 0 else _MOCK_FPS_DEFAULT
        resolved_duration_ms = (
            metadata.duration_ms if metadata.duration_ms > 0 else _MOCK_DURATION_MS_DEFAULT
        )
        return resolved_fps, resolved_duration_ms
    except Exception:
        return _MOCK_FPS_DEFAULT, _MOCK_DURATION_MS_DEFAULT


@register_pose_estimator("mock")
class MockPoseEstimator:
    """Deterministic mock body-pose estimator. NOT a trained model."""

    model_name = "mock-pose-estimator"
    model_version = "0.0.0-mock"

    def predict(
        self,
        video_path: Path,
        *,
        duration_ms: int | None = None,
        fps: float | None = None,
    ) -> PoseSequence:
        seed = _stable_seed("pose", str(video_path))
        resolved_fps, resolved_duration_ms = _mock_video_timeline(
            video_path, duration_ms=duration_ms, fps=fps
        )
        n_frames = max(1, int(resolved_duration_ms / 1000 * resolved_fps))
        frames = []
        for frame_idx in range(n_frames):
            frame_ms = int(frame_idx / resolved_fps * 1000)
            jitter = _deterministic_unit_floats(seed + frame_idx, len(_POSE_LANDMARK_NAMES))
            keypoints = tuple(
                Keypoint(name=name, x=j, y=1.0 - j, z=0.0, visibility=0.9)
                for name, j in zip(_POSE_LANDMARK_NAMES, jitter, strict=True)
            )
            frames.append(PoseFrame(frame_ms=frame_ms, keypoints=keypoints, confidence=0.6))
        return PoseSequence(
            frames=tuple(frames),
            model_name=self.model_name,
            model_version=self.model_version,
            fps=resolved_fps,
        )


@register_hand_estimator("mock")
class MockHandEstimator:
    """Deterministic mock hand-landmark estimator. NOT a trained model."""

    model_name = "mock-hand-estimator"
    model_version = "0.0.0-mock"

    def predict(
        self,
        video_path: Path,
        *,
        duration_ms: int | None = None,
        fps: float | None = None,
    ) -> HandSequence:
        seed = _stable_seed("hands", str(video_path))
        resolved_fps, resolved_duration_ms = _mock_video_timeline(
            video_path, duration_ms=duration_ms, fps=fps
        )
        n_frames = max(1, int(resolved_duration_ms / 1000 * resolved_fps))
        frames = []
        for frame_idx in range(n_frames):
            frame_ms = int(frame_idx / resolved_fps * 1000)
            for handedness in ("left", "right"):
                jitter = _deterministic_unit_floats(
                    seed + frame_idx + (1 if handedness == "right" else 0),
                    len(_HAND_LANDMARK_NAMES),
                )
                keypoints = tuple(
                    Keypoint(name=name, x=j, y=1.0 - j, z=0.0, visibility=0.85)
                    for name, j in zip(_HAND_LANDMARK_NAMES, jitter, strict=True)
                )
                frames.append(
                    HandFrame(
                        frame_ms=frame_ms,
                        handedness=handedness,
                        keypoints=keypoints,
                        confidence=0.6,
                    )
                )
        return HandSequence(
            frames=tuple(frames),
            model_name=self.model_name,
            model_version=self.model_version,
            fps=resolved_fps,
        )


@register_yoyo_detector("mock")
class MockYoyoDetector:
    """Deterministic mock yo-yo object detector.

    No real yo-yo detection weights are available yet; this exists purely so
    the rest of the pipeline (tracking, feature extraction, event detection,
    scoring, review UI) can be exercised end to end.
    """

    model_name = "mock-yoyo-detector"
    model_version = "0.0.0-mock"

    def predict(self, frame_batch: list[FrameRef]) -> list[Detection]:
        detections = []
        for frame in frame_batch:
            seed = _stable_seed("yoyo", str(frame.frame_ms))
            jitter = _deterministic_unit_floats(seed, 4)
            bbox = BoundingBox(
                x=0.4 + jitter[0] * 0.1,
                y=0.4 + jitter[1] * 0.1,
                width=0.05 + jitter[2] * 0.02,
                height=0.05 + jitter[3] * 0.02,
            )
            detections.append(
                Detection(
                    frame_ms=frame.frame_ms,
                    bbox=bbox,
                    confidence=0.5 + jitter[0] * 0.3,
                    class_label="yoyo",
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
            )
        return detections


@register_tracker("mock")
class MockTracker:
    """Deterministic mock single-object tracker (assumes one visible yo-yo)."""

    model_name = "mock-tracker"
    model_version = "0.0.0-mock"

    def __init__(self) -> None:
        self._track_id = "track-0"

    def reset(self) -> None:
        self._track_id = "track-0"

    def update(self, detections: list[Detection], timestamp_ms: int) -> list[Track]:
        tracks = []
        for detection in detections:
            if detection.frame_ms != timestamp_ms:
                continue
            tracks.append(
                Track(
                    track_id=self._track_id,
                    frame_ms=detection.frame_ms,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_label=detection.class_label,
                    visibility=VisibilityState.VISIBLE,
                    interpolated=False,
                )
            )
        return tracks


@register_temporal_event_detector("mock")
class MockTemporalEventDetector:
    """Deterministic mock temporal segmentation across the 23 supported families.

    No real temporal event detection model is trained yet. This produces a
    plausible, varied, fully-labelled mock timeline (spanning the full
    feature-set duration) purely so the review UI, scoring engine, and
    exports have real end-to-end data to operate on.
    """

    model_name = "mock-temporal-event-detector"
    model_version = "0.0.0-mock"

    def predict(
        self, features: FeatureSet
    ) -> tuple[list[AnalysisEventPrediction], list[DeductionPrediction]]:
        if not features.frames:
            return [], []

        total_ms = features.frames[-1].frame_ms
        seed = _stable_seed("temporal-events", str(total_ms), str(len(features.frames)))
        # ~1 event every 2.5s across the full clip, capped to keep review manageable.
        n_events = max(3, min(total_ms // 2500, 150))
        events: list[AnalysisEventPrediction] = []

        for i in range(n_events):
            jitter = _deterministic_unit_floats(seed + i, 3)
            family = _EVENT_FAMILY_CYCLE[(seed + i) % len(_EVENT_FAMILY_CYCLE)]
            outcome = _OUTCOME_CYCLE[(seed + i) % len(_OUTCOME_CYCLE)]
            band = _BAND_CYCLE[(seed + i) % len(_BAND_CYCLE)]
            if n_events == 1:
                start_ms = 0
            else:
                start_ms = int((i / (n_events - 1)) * max(0, total_ms - 800))
            end_ms = min(total_ms, start_ms + 400 + int(jitter[0] * 600))
            confidence = round(0.4 + jitter[1] * 0.55, 3)

            events.append(
                AnalysisEventPrediction(
                    label=f"{family.value}_{i}",
                    family=family,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=confidence,
                    outcome=outcome,
                    difficulty_band=band,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    evidence=(
                        EvidenceRef(
                            frame_ms=start_ms,
                            note="mock evidence: synthetic feature-window heuristic, not measured",
                        ),
                    ),
                )
            )

        deductions: list[DeductionPrediction] = []
        if n_events >= 6:
            jitter = _deterministic_unit_floats(seed + 999, 2)
            deductions.append(
                DeductionPrediction(
                    type=DeductionType.YOYO_STOP,
                    timestamp_ms=int(total_ms * 0.5),
                    quantity=1,
                    confidence=round(0.5 + jitter[0] * 0.4, 3),
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
            )

        return events, deductions
