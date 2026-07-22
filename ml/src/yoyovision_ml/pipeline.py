"""End-to-end analysis pipeline orchestration.

Wires together preprocessing, pose/hand/yo-yo/tracking adapters (resolved by
name through `adapters_registry` -- product principle #5: replaceable
adapters), string/feature analysis, temporal event detection, and the
deterministic scoring engine. This module has no database or web-framework
dependency: it is pure, given a video path and metadata, so it can run
identically inside a Celery worker or a test.

Adapter names default to `"mock"`, the deterministic mock adapters that ship
today (product principle #7). Swapping to real weights later is a config
change (pass different `*_adapter_name` values plus `adapter_kwargs`), not a
code change.

Prompt F (production inference) additions, all optional/backward-compatible
so every existing call site (including the mock-only test suite) keeps
working unchanged:

* `adapter_kwargs` -- per-adapter constructor kwargs (e.g. `weights_path`,
  `device`), forwarded to `adapters_registry.create_*`.
* `device_preference` -- resolved once via `inference.device.resolve_device`
  and recorded on `PipelineResult`; adapters that accept a `device` kwarg
  get the resolved value merged in automatically.
* `model_registry` -- an `inference.model_registry.ModelRegistry` used to
  load each adapter once per worker process instead of once per job.
* `cancellation` -- an `inference.cancellation.CancellationToken` polled at
  every stage boundary, so a human "cancel" request or a wall-clock timeout
  stops the run promptly instead of finishing an unwanted job.
* `stage_callback` -- invoked after every stage with `(stage, elapsed_ms)`,
  letting the worker persist live progress without this module knowing
  about Celery or SQLAlchemy.
* `reference_baseline` -- optional monitoring baseline; when given,
  `PipelineResult.monitoring` includes class/confidence drift scores.

`PipelineResult` gained `stage_durations_ms`, `device`, `runtime_versions`,
and `monitoring` fields -- all with safe defaults, so existing
`PipelineResult(events=..., deductions=..., score=..., model_versions=...)`
construction and existing attribute access both keep working.

Prompt E (RGB, string, and audio fusion) addition, also optional/
backward-compatible: `feature_fusion_mode` defaults to `"kinematics_only"`
(today's exact pre-Prompt-E behavior). Passing `"fused"` additionally runs
`RgbEncoder`/`StringSegmenter`/`AudioAnalyzer` adapters (see
`yoyovision_ml.multimodal`) during the feature-extraction stage and merges
their output into the feature timeline via
`multimodal.features.fuse_feature_sets` -- their resolved `model_versions`
entries (`rgb_encoder`, `string_segmenter`, `audio_analyzer`) only appear
when fusion actually ran, so `"kinematics_only"` callers see an unchanged
`model_versions` dict shape.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

#: Importing this module registers the "mock" adapters with `adapters_registry`.
from yoyovision_ml import adapters_mock  # noqa: F401,E402
from yoyovision_ml.adapters_registry import (
    create_audio_analyzer,
    create_hand_estimator,
    create_pose_estimator,
    create_rgb_encoder,
    create_string_segmenter,
    create_temporal_event_detector,
    create_tracker,
    create_yoyo_detector,
)
from yoyovision_ml.domain import (
    AnalysisEventPrediction,
    DeductionPrediction,
    PipelineStage,
    ScoreBreakdown,
    Track,
)
from yoyovision_ml.feature_extraction import DeterministicFeatureExtractor
from yoyovision_ml.inference.cancellation import CancellationToken
from yoyovision_ml.inference.device import resolve_device
from yoyovision_ml.inference.device import runtime_versions as _runtime_versions
from yoyovision_ml.inference.model_registry import ModelArtifactSpec, ModelRegistry
from yoyovision_ml.inference.monitoring import (
    MonitoringSignals,
    ReferenceBaseline,
    compute_monitoring_signals,
)
from yoyovision_ml.inference.timing import StageTimings
from yoyovision_ml.interfaces import (
    AudioAnalyzer,
    HandEstimator,
    ObjectTracker,
    PoseEstimator,
    RgbEncoder,
    StringSegmenter,
    TemporalEventDetector,
    YoyoDetector,
)

#: Importing this module registers Prompt E's "mock" rgb/string-seg/audio
#: adapters with `adapters_registry`.
from yoyovision_ml.multimodal import adapters_mock as multimodal_adapters_mock  # noqa: F401,E402
from yoyovision_ml.multimodal.features import fuse_feature_sets
from yoyovision_ml.preprocessing import extract_frames
from yoyovision_ml.ruleset import Ruleset
from yoyovision_ml.scoring_engine import DeterministicScoringEngine
from yoyovision_ml.string_analysis import DeterministicStringAnalyzer

#: Adapter roles keyed the same way `adapter_kwargs` is, e.g.
#: `adapter_kwargs={"yoyo": {"weights_path": "..."}}`. The last three (Prompt
#: E) are only ever resolved when `feature_fusion_mode="fused"`.
_ADAPTER_ROLES = (
    "pose",
    "hand",
    "yoyo",
    "tracker",
    "temporal_event",
    "rgb",
    "string_seg",
    "audio",
)

#: Prompt E: whether `run_analysis_pipeline` runs the additional RGB/string-
#: segmentation/audio adapters and fuses their output into the feature
#: timeline. Defaults to `"kinematics_only"` -- today's exact pre-Prompt-E
#: behavior -- so every existing call site (including the mock-only test
#: suite) keeps working unchanged; `"fused"` is opt-in.
FeatureFusionMode = Literal["kinematics_only", "fused"]

StageCallback = Callable[[PipelineStage, float], None]


@dataclass(slots=True, frozen=True)
class PipelineResult:
    events: list[AnalysisEventPrediction]
    deductions: list[DeductionPrediction]
    score: ScoreBreakdown
    model_versions: dict[str, str]
    stage_durations_ms: dict[str, float] = field(default_factory=dict)
    device: str = "cpu"
    runtime_versions: dict[str, str] = field(default_factory=dict)
    monitoring: MonitoringSignals | None = None


def _create_adapter(
    role: str,
    factory: Callable[..., object],
    name: str,
    kwargs: dict[str, object],
    device: str,
    registry: ModelRegistry | None,
) -> object:
    """Resolves one adapter by name, merging in the pipeline's resolved
    `device` (only adapters that don't already specify one get it) and, if a
    `ModelRegistry` was supplied, loading it through that registry's
    load-once-per-process cache instead of constructing it fresh every call.
    """
    merged_kwargs = dict(kwargs)
    if "device" not in merged_kwargs and name != "mock":
        merged_kwargs["device"] = device

    if registry is None:
        return factory(name, **merged_kwargs)

    expected_sha256 = merged_kwargs.pop("expected_sha256", None)
    weights_path = merged_kwargs.get("weights_path") or merged_kwargs.get("model_path")
    cache_key = f"{role}:{name}:{sorted(merged_kwargs.items())}"
    spec = ModelArtifactSpec(
        name=f"{role}:{name}",
        version=name,
        path=Path(str(weights_path)) if weights_path else None,
        expected_sha256=expected_sha256,  # type: ignore[arg-type]
        device=device,
    )
    loaded = registry.get_or_load(cache_key, lambda: factory(name, **merged_kwargs), spec=spec)
    return loaded.instance


def run_analysis_pipeline(
    video_path: Path,
    duration_ms: int,
    fps: float,
    ruleset: Ruleset,
    freestyle_evaluation: None = None,
    pose_adapter_name: str = "mock",
    hand_adapter_name: str = "mock",
    yoyo_adapter_name: str = "mock",
    tracker_adapter_name: str = "mock",
    temporal_event_adapter_name: str = "mock",
    sample_fps: float = 15.0,
    adapter_kwargs: Mapping[str, Mapping[str, object]] | None = None,
    device_preference: str = "cpu",
    model_registry: ModelRegistry | None = None,
    cancellation: CancellationToken | None = None,
    stage_callback: StageCallback | None = None,
    reference_baseline: ReferenceBaseline | None = None,
    feature_fusion_mode: FeatureFusionMode = "kinematics_only",
    rgb_adapter_name: str = "mock",
    string_seg_adapter_name: str = "mock",
    audio_adapter_name: str = "mock",
) -> PipelineResult:
    """Runs the full detection + scoring pipeline for one video.

    `freestyle_evaluation` is intentionally `None` here: per MVP scope,
    Freestyle Evaluation is a manual human entry made later during review,
    never a pipeline output (see `docs/ruleset.md`).
    """
    cancellation = cancellation or CancellationToken()
    timings = StageTimings()
    device_info = resolve_device(device_preference)
    kwargs_by_role: dict[str, dict[str, object]] = {
        role: dict(adapter_kwargs.get(role, {})) if adapter_kwargs else {}
        for role in _ADAPTER_ROLES
    }

    def _stage(stage: PipelineStage) -> None:
        cancellation.check(stage.value)

    def _finish(stage: PipelineStage) -> None:
        if stage_callback is not None:
            stage_callback(stage, timings.durations_ms.get(stage.value, 0.0))

    _stage(PipelineStage.POSE_EXTRACTION)
    with timings.measure(PipelineStage.POSE_EXTRACTION.value):
        pose_estimator: PoseEstimator = _create_adapter(  # type: ignore[assignment]
            "pose",
            create_pose_estimator,
            pose_adapter_name,
            kwargs_by_role["pose"],
            device_info.resolved,
            model_registry,
        )
        pose_sequence = pose_estimator.predict(video_path, duration_ms=duration_ms, fps=fps)
    _finish(PipelineStage.POSE_EXTRACTION)

    _stage(PipelineStage.HAND_EXTRACTION)
    with timings.measure(PipelineStage.HAND_EXTRACTION.value):
        hand_estimator: HandEstimator = _create_adapter(  # type: ignore[assignment]
            "hand",
            create_hand_estimator,
            hand_adapter_name,
            kwargs_by_role["hand"],
            device_info.resolved,
            model_registry,
        )
        hand_sequence = hand_estimator.predict(video_path, duration_ms=duration_ms, fps=fps)
    _finish(PipelineStage.HAND_EXTRACTION)

    _stage(PipelineStage.YOYO_DETECTION)
    with timings.measure(PipelineStage.YOYO_DETECTION.value):
        yoyo_detector: YoyoDetector = _create_adapter(  # type: ignore[assignment]
            "yoyo",
            create_yoyo_detector,
            yoyo_adapter_name,
            kwargs_by_role["yoyo"],
            device_info.resolved,
            model_registry,
        )
        frames = extract_frames(video_path, duration_ms=duration_ms, fps=fps, sample_fps=sample_fps)
        detections = yoyo_detector.predict(frames)
    _finish(PipelineStage.YOYO_DETECTION)

    _stage(PipelineStage.TRACKING)
    with timings.measure(PipelineStage.TRACKING.value):
        tracker: ObjectTracker = _create_adapter(  # type: ignore[assignment]
            "tracker",
            create_tracker,
            tracker_adapter_name,
            kwargs_by_role["tracker"],
            device_info.resolved,
            model_registry,
        )
        tracker.reset()
        tracks: list[Track] = [
            track
            for frame in frames
            for track in tracker.update(detections, timestamp_ms=frame.frame_ms)
        ]
    _finish(PipelineStage.TRACKING)

    _stage(PipelineStage.STRING_ANALYSIS)
    with timings.measure(PipelineStage.STRING_ANALYSIS.value):
        string_features = DeterministicStringAnalyzer().analyze(tracks, hand_sequence)
    _finish(PipelineStage.STRING_ANALYSIS)

    _stage(PipelineStage.FEATURE_EXTRACTION)
    multimodal_model_versions: dict[str, str] = {}
    with timings.measure(PipelineStage.FEATURE_EXTRACTION.value):
        features = DeterministicFeatureExtractor().extract(
            pose_sequence, hand_sequence, tracks, string_features
        )
        if feature_fusion_mode == "fused":
            rgb_encoder: RgbEncoder = _create_adapter(  # type: ignore[assignment]
                "rgb",
                create_rgb_encoder,
                rgb_adapter_name,
                kwargs_by_role["rgb"],
                device_info.resolved,
                model_registry,
            )
            string_segmenter: StringSegmenter = _create_adapter(  # type: ignore[assignment]
                "string_seg",
                create_string_segmenter,
                string_seg_adapter_name,
                kwargs_by_role["string_seg"],
                device_info.resolved,
                model_registry,
            )
            audio_analyzer: AudioAnalyzer = _create_adapter(  # type: ignore[assignment]
                "audio",
                create_audio_analyzer,
                audio_adapter_name,
                kwargs_by_role["audio"],
                device_info.resolved,
                model_registry,
            )
            rgb_features = rgb_encoder.encode(frames)
            string_seg_features = string_segmenter.segment(frames, tracks)
            audio_features = audio_analyzer.analyze(video_path, duration_ms)
            features = fuse_feature_sets(
                features, rgb_features, string_seg_features, audio_features
            )
            multimodal_model_versions = {
                "rgb_encoder": f"{rgb_encoder.model_name}@{rgb_encoder.model_version}",
                "string_segmenter": (
                    f"{string_segmenter.model_name}@{string_segmenter.model_version}"
                ),
                "audio_analyzer": f"{audio_analyzer.model_name}@{audio_analyzer.model_version}",
            }
    _finish(PipelineStage.FEATURE_EXTRACTION)

    _stage(PipelineStage.TEMPORAL_EVENT_DETECTION)
    with timings.measure(PipelineStage.TEMPORAL_EVENT_DETECTION.value):
        temporal_event_detector: TemporalEventDetector = _create_adapter(  # type: ignore[assignment]
            "temporal_event",
            create_temporal_event_detector,
            temporal_event_adapter_name,
            kwargs_by_role["temporal_event"],
            device_info.resolved,
            model_registry,
        )
        events, deductions = temporal_event_detector.predict(features)
    _finish(PipelineStage.TEMPORAL_EVENT_DETECTION)

    _stage(PipelineStage.SCORING)
    with timings.measure(PipelineStage.SCORING.value):
        score = DeterministicScoringEngine().calculate(
            events=events,
            deductions=deductions,
            freestyle_evaluation=freestyle_evaluation,
            ruleset=ruleset,
        )
    _finish(PipelineStage.SCORING)

    monitoring = compute_monitoring_signals(events, tracks, reference=reference_baseline)

    return PipelineResult(
        events=events,
        deductions=deductions,
        score=score,
        model_versions={
            "pose_estimator": f"{pose_estimator.model_name}@{pose_estimator.model_version}",
            "hand_estimator": f"{hand_estimator.model_name}@{hand_estimator.model_version}",
            "yoyo_detector": f"{yoyo_detector.model_name}@{yoyo_detector.model_version}",
            "tracker": f"{tracker.model_name}@{tracker.model_version}",
            "temporal_event_detector": (
                f"{temporal_event_detector.model_name}@{temporal_event_detector.model_version}"
            ),
            **multimodal_model_versions,
        },
        stage_durations_ms=timings.durations_ms,
        device=device_info.resolved,
        runtime_versions=_runtime_versions(),
        monitoring=monitoring,
    )
