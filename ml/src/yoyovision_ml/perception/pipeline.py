"""`PerceptionPipeline`: orchestrates pose/hand/yo-yo/tracker adapters +
kinematic feature computation + artefact writing for one video (Prompt B).

Deliberately separate from `yoyovision_ml.pipeline.run_analysis_pipeline`
(the existing scoring-oriented pipeline): this one stops at the feature
artefact and never touches temporal event detection or scoring. Wiring a
perception artefact into the full analysis pipeline is explicitly Prompt F's
job ("Integrate validated models into the existing YoYoVision worker"), not
this one's.

Adapters are resolved by name through `adapters_registry` (product principle
#5), exactly like the existing pipeline -- swapping `"mock"` for
`"mediapipe"`/`"pytorch"`/`"onnx"`/`"kalman"` is a configuration change here
too. Importing this module (via `yoyovision_ml.perception`) registers all of
those real adapters alongside the `"mock"` ones from `adapters_mock`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yoyovision_ml import (
    adapters_mock,  # noqa: F401 -- registers "mock" adapters
    perception,  # noqa: F401 -- registers real adapters (this package)
)
from yoyovision_ml.adapters_registry import (
    create_hand_estimator,
    create_pose_estimator,
    create_tracker,
    create_yoyo_detector,
)
from yoyovision_ml.domain import Detection, FeatureSet, HandSequence, PoseSequence, Track
from yoyovision_ml.interfaces import HandEstimator, ObjectTracker, PoseEstimator, YoyoDetector
from yoyovision_ml.perception.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    PerceptionMetadata,
    compute_video_checksum,
    write_artifact,
)
from yoyovision_ml.perception.features import compute_kinematic_features
from yoyovision_ml.preprocessing import extract_frames

#: Bumped whenever frame sampling / preprocessing semantics change in a way
#: that would make an old artefact non-reproducible from the current code.
PREPROCESSING_VERSION = "0.1.0"


@dataclass(slots=True, frozen=True)
class PerceptionResult:
    pose_sequence: PoseSequence
    hand_sequence: HandSequence
    yoyo_detections: tuple[Detection, ...]
    yoyo_tracks: tuple[Track, ...]
    feature_set: FeatureSet
    metadata: PerceptionMetadata


class PerceptionPipeline:
    """Runs pose/hand/yo-yo detection + tracking + feature extraction for one video.

    `*_adapter_kwargs` are forwarded verbatim to the resolved adapter's
    constructor (e.g. `yoyo_adapter_kwargs={"weights_path": "..."}` for the
    `"pytorch"`/`"onnx"` detectors, or `tracker_adapter_kwargs={"max_gap_ms":
    300, "static_camera": True}` for `"kalman"`).
    """

    def __init__(
        self,
        pose_adapter_name: str = "mock",
        hand_adapter_name: str = "mock",
        yoyo_adapter_name: str = "mock",
        tracker_adapter_name: str = "mock",
        pose_adapter_kwargs: dict[str, object] | None = None,
        hand_adapter_kwargs: dict[str, object] | None = None,
        yoyo_adapter_kwargs: dict[str, object] | None = None,
        tracker_adapter_kwargs: dict[str, object] | None = None,
        sample_fps: float = 15.0,
    ) -> None:
        self.sample_fps = sample_fps
        self._pose_estimator: PoseEstimator = create_pose_estimator(  # type: ignore[assignment]
            pose_adapter_name, **(pose_adapter_kwargs or {})
        )
        self._hand_estimator: HandEstimator = create_hand_estimator(  # type: ignore[assignment]
            hand_adapter_name, **(hand_adapter_kwargs or {})
        )
        self._yoyo_detector: YoyoDetector = create_yoyo_detector(  # type: ignore[assignment]
            yoyo_adapter_name, **(yoyo_adapter_kwargs or {})
        )
        self._tracker: ObjectTracker = create_tracker(  # type: ignore[assignment]
            tracker_adapter_name, **(tracker_adapter_kwargs or {})
        )

    def run(self, video_path: Path, duration_ms: int, fps: float) -> PerceptionResult:
        pose_sequence = self._pose_estimator.predict(video_path)
        hand_sequence = self._hand_estimator.predict(video_path)

        frames = extract_frames(
            video_path, duration_ms=duration_ms, fps=fps, sample_fps=self.sample_fps
        )
        detections = self._yoyo_detector.predict(frames)

        self._tracker.reset()
        tracks = [
            track
            for frame in frames
            for track in self._tracker.update(detections, timestamp_ms=frame.frame_ms)
        ]

        feature_set = compute_kinematic_features(
            pose_sequence, hand_sequence, tracks, fps=self.sample_fps
        )

        track_quality_fn = getattr(self._tracker, "track_quality", None)
        model_versions = {
            "pose_estimator": (
                f"{self._pose_estimator.model_name}@{self._pose_estimator.model_version}"
            ),
            "hand_estimator": (
                f"{self._hand_estimator.model_name}@{self._hand_estimator.model_version}"
            ),
            "yoyo_detector": (
                f"{self._yoyo_detector.model_name}@{self._yoyo_detector.model_version}"
            ),
            "tracker": f"{self._tracker.model_name}@{self._tracker.model_version}",
        }
        metadata = PerceptionMetadata(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            video_filename=video_path.name,
            video_checksum_sha256=(
                compute_video_checksum(video_path) if video_path.exists() else "unavailable"
            ),
            duration_ms=duration_ms,
            source_fps=fps,
            processed_fps=self.sample_fps,
            frame_count=len(feature_set.frames),
            preprocessing_version=PREPROCESSING_VERSION,
            model_versions=model_versions,
            feature_names=feature_set.feature_names,
            track_quality=track_quality_fn() if callable(track_quality_fn) else None,
        )

        return PerceptionResult(
            pose_sequence=pose_sequence,
            hand_sequence=hand_sequence,
            yoyo_detections=tuple(detections),
            yoyo_tracks=tuple(tracks),
            feature_set=feature_set,
            metadata=metadata,
        )

    def run_and_write(
        self, video_path: Path, duration_ms: int, fps: float, output_dir: Path, name: str
    ) -> tuple[PerceptionResult, Path, Path]:
        result = self.run(video_path, duration_ms=duration_ms, fps=fps)
        parquet_path, metadata_path = write_artifact(
            result.feature_set, result.metadata, output_dir, name
        )
        return result, parquet_path, metadata_path
