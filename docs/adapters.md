# ML Adapters

## Status today: mock by default, real adapters available for perception and temporal events

**No trained yo-yo detector checkpoint ships with this repository, and the
default pipeline's temporal-event-detection stage is still mock-only.**
Every adapter registered under the name `"mock"` in
`ml/src/yoyovision_ml/adapters_mock.py` is a deterministic function of its
input (a stable SHA-256-derived seed feeding a tiny xorshift PRNG — see
`_stable_seed`/`_deterministic_unit_floats`), **not** a trained model. This
is intentional (product principles #6 and #7 — "never fabricate model
accuracy, trained weights or inference results" and "use a deterministic
mock inference adapter until genuine model weights are supplied") and is
enforced by convention, not just documentation:

The **perception pipeline** (`ml/src/yoyovision_ml/perception/`, see below)
adds real, non-mock pose/hand/yo-yo/tracker adapters alongside the mocks —
pose/hand estimation via MediaPipe is genuinely working today (bundled
pretrained models); the yo-yo detector's PyTorch/ONNX adapters are real,
replaceable inference *slots* that refuse to run until a checkpoint is
explicitly configured (see "Perception adapters" below), since no trained
yo-yo detector weights exist yet.

The **temporal trick-event model** (`ml/src/yoyovision_ml/events/`, see
"Temporal event detector adapters" below) similarly adds a real, trainable
`TemporalEventDetector` adapter (`"torch"`) alongside two always-available,
never-trained reference baselines (`"majority"`, `"rules"`). Unlike the
perception adapters, this one *can* be trained end to end within this
repository today — but only on the package's deterministic synthetic
dataset, since no real annotated 1A footage exists yet. Its checkpoints are
real, loadable, runnable model weights; they are just not weights that have
seen real footage, which is why the default pipeline still points at the
mock adapter rather than this one.

- Every mock adapter's `model_name` is prefixed `mock-` (e.g.
  `mock-pose-estimator`, `mock-yoyo-detector`, `mock-temporal-event-detector`).
- Every mock adapter's `model_version` is literally `"0.0.0-mock"`.
- The `MockTemporalEventDetector`'s synthetic evidence notes say outright:
  `"mock evidence: synthetic feature-window heuristic, not measured"`.
- These strings flow untouched into `AnalysisEvent.model_name` /
  `model_version` and `evidence_json`, so a reviewer looking at any event in
  the UI or export can immediately tell it came from a mock, not a real
  detector.

Same-input-same-output determinism means pipeline runs, tests, and demos
are fully reproducible without depending on any ML framework being
installed with real weights.

## Why Protocols, not concrete classes

`ml/src/yoyovision_ml/interfaces.py` defines a `Protocol` for every
replaceable component. Calling code (the `workers` pipeline) only ever
types against these Protocols:

| Protocol | Method | Purpose |
| --- | --- | --- |
| `PoseEstimator` | `predict(video_path: Path) -> PoseSequence` | Full-body landmarks across a video |
| `HandEstimator` | `predict(video_path: Path) -> HandSequence` | Both-hand landmarks across a video |
| `YoyoDetector` | `predict(frame_batch: list[FrameRef]) -> list[Detection]` | Per-frame yo-yo bounding-box detection |
| `ObjectTracker` | `update(detections, timestamp_ms) -> list[Track]`, `reset()` | Associates per-frame detections into persistent tracks |
| `StringAnalyzer` | `analyze(yoyo_track, hand_sequence) -> FeatureSet` | String/slack geometry from tracked positions |
| `FeatureExtractor` | `extract(pose_sequence, hand_sequence, yoyo_tracks, string_features) -> FeatureSet` | Unified per-frame feature vector |
| `TemporalEventDetector` | `predict(features) -> tuple[list[AnalysisEventPrediction], list[DeductionPrediction]]` | Segments the feature timeline into atomic events + equipment deductions |
| `ScoringEngine` | `calculate(events, deductions, freestyle_evaluation, ruleset) -> ScoreBreakdown` | Deterministic rules-based scoring (see `docs/ruleset.md`) |
| `StoragePort` | `put`/`get`/`delete`/`exists`/`signed_url` | Local-filesystem vs. S3-compatible storage |

All are `@runtime_checkable`, so `isinstance(adapter, PoseEstimator)` works
for a quick sanity check in tests, but static typing (mypy) is the primary
enforcement mechanism — a class that doesn't fully implement a Protocol's
methods/attributes fails a typed call site at type-check time, not at
runtime.

`PoseEstimator`/`HandEstimator`/`YoyoDetector`/`ObjectTracker`/
`StringAnalyzer`/`TemporalEventDetector` all additionally require
`model_name: str` and `model_version: str` instance attributes — every
adapter is required, by the type itself, to be able to say what produced
its output (product principle #2).

## The registry: swapping adapters without touching calling code

`ml/src/yoyovision_ml/adapters_registry.py` holds one name→factory dict per
component kind (`_POSE_ESTIMATORS`, `_HAND_ESTIMATORS`, `_YOYO_DETECTORS`,
`_TRACKERS`, `_TEMPORAL_EVENT_DETECTORS`, `_STORAGE_BACKENDS`). A concrete
adapter registers itself with a decorator at import time:

```python
@register_yoyo_detector("mock")
class MockYoyoDetector:
    model_name = "mock-yoyo-detector"
    model_version = "0.0.0-mock"

    def predict(self, frame_batch: list[FrameRef]) -> list[Detection]:
        ...
```

Calling code never imports `MockYoyoDetector` directly — it resolves the
adapter by a config-driven name string:

```python
detector = create_yoyo_detector(settings.yoyo_detector_name)  # e.g. "mock" today
```

`create_*` raises `AdapterNotRegisteredError` (listing every name that *is*
registered) if the configured name has no factory — a fail-fast, explicit
error rather than a silent fallback to some default behavior. `api/main.py`
and `workers` import `yoyovision_ml.adapters_mock` (and `.storage`) purely
for their registration side effects (`# noqa: F401`), which is the only
place the mock implementations are referenced by name anywhere in `api` or
`workers`.

## Adding a real adapter

To replace the mock yo-yo detector with a real MediaPipe/PyTorch-backed
one, for example:

1. Implement a class satisfying the `YoyoDetector` Protocol (same
   `predict(frame_batch) -> list[Detection]` signature, real `model_name`
   and a real, truthful `model_version` — never claim mock-level certainty
   or fabricate an accuracy figure that hasn't actually been measured).
2. Register it: `@register_yoyo_detector("mediapipe-yoyo-v1")` in a new
   module (e.g. `adapters_mediapipe.py`), imported for its side effect
   wherever the mock adapters currently are.
3. Point configuration (`YOYO_DETECTOR_NAME=mediapipe-yoyo-v1` or
   equivalent settings field) at the new name. No changes to `workers`'
   pipeline orchestration, `api`, or the frontend are required — they only
   ever depend on the `YoyoDetector` Protocol and the config string.
4. Keep the mock adapter registered and available (e.g. for CI, offline
   demos, or environments without model weights available) — swapping the
   *default* config value doesn't have to mean deleting the mock.

The same four-step process applies to `PoseEstimator`, `HandEstimator`,
`ObjectTracker`, `StringAnalyzer`, and `TemporalEventDetector`. `ScoringEngine`
is deliberately the one component that should almost never be swapped for a
learned model — see `docs/ruleset.md` for why scoring stays rules-based.

## Storage adapters (`storage.py`)

`StoragePort` has two production-ready implementations today (not mocks —
these do real file I/O):

- `LocalFilesystemStorage` (`register_storage_backend("local")`) — for
  development. Every `storage_key` is validated against path traversal
  (`_assert_safe_relative_key` rejects absolute paths, `..` segments, and
  any resolved path escaping the storage root) before any filesystem
  operation. Its `signed_url()` has no real signed-URL mechanism (there's
  no CDN/object store to sign against locally), so it deliberately returns
  an authenticated-proxy path instead of a filesystem path, keeping local
  dev's security model consistent with production's "never expose storage
  paths" requirement.
- `S3CompatibleStorage` (`register_storage_backend("s3")`) — for
  production, also usable locally against MinIO for parity testing. Lazily
  imports `boto3` (raising a clear `RuntimeError` naming the missing
  dependency if it's absent, rather than failing at module import time for
  everyone). `signed_url()` returns a real, time-limited presigned S3 URL.

Selection is via `STORAGE_BACKEND=local|s3` config, resolved through
`create_storage_backend()` — the same registry pattern as the ML adapters,
so `api` and `workers` never import either storage class directly.

## Perception adapters (`ml/src/yoyovision_ml/perception/`)

A second, standalone pipeline (`perception/pipeline.py`'s `PerceptionPipeline`
— deliberately separate from `pipeline.run_analysis_pipeline`; it stops at a
feature artifact and never touches scoring) that resolves the same kind of
Protocol/registry-based adapters, plus a `KalmanYoyoTracker` and a set of
real pose/hand/yo-yo adapters:

| Name | Registered as | Requires | Behavior when unconfigured/missing |
| --- | --- | --- | --- |
| `MediaPipePoseEstimator` / `MediaPipeHandEstimator` | `"mediapipe"` (pose/hand) | `pip install 'yoyovision-ml[mediapipe]'` | Raises `MissingOptionalDependencyError` naming `mediapipe`/`cv2` and the extra to install |
| `PyTorchYoyoDetector` | `"pytorch"` (yo-yo detector) | `pip install 'yoyovision-ml[torch]'` **and** a configured checkpoint | Raises `ModelWeightsNotConfiguredError` if no `weights_path`/`YOYOVISION_TORCH_YOYO_WEIGHTS` is set or the path doesn't exist; raises `MissingOptionalDependencyError` if `torch` itself is absent |
| `ONNXYoyoDetector` | `"onnx"` (yo-yo detector) | `pip install 'yoyovision-ml[onnx]'` **and** a configured `.onnx` model | Same two-stage fail-clear behavior as the PyTorch adapter (`model_path`/`YOYOVISION_ONNX_YOYO_MODEL`) |
| `KalmanYoyoTracker` | `"kalman"` (tracker) | none (hand-written 4-state Kalman filter, no numpy/scipy) | N/A — always available; `max_gap_ms`/`static_camera`/`process_noise` are constructor kwargs |

`MissingOptionalDependencyError` and `ModelWeightsNotConfiguredError`
(`perception/errors.py`) are the two failure modes every real perception
adapter uses — deliberately never a silent fallback to mock output, per the
same "never fabricate model accuracy, trained weights or inference results"
principle as above. The PyTorch adapter also loads checkpoints with
`torch.load(..., weights_only=True)`, since a checkpoint file is untrusted
input.

Adapter construction kwargs flow through the registry's `create_*(name,
**kwargs)` (e.g. `create_yoyo_detector("pytorch", weights_path=...)`), and
`PerceptionPipeline.__init__`'s `*_adapter_kwargs` parameters forward
directly to whichever adapter is selected — see `perception/cli.py`'s
`--yoyo-weights`/`--tracker-max-gap-ms`/`--static-camera` flags for the
end-to-end wiring.

### Feature computation, artifacts, and evaluation

- `perception/features.py`'s `compute_kinematic_features` turns
  `PoseSequence`/`HandSequence`/`list[Track]` into a `FeatureSet` keyed to
  each yo-yo track frame's real timestamp (positions, relative wrist
  offsets, velocity/acceleration/direction, hand distance, elbow angles,
  shoulder width, body-centered "stage position", plus the track's
  confidence/visibility/interpolated flags). Both mock and MediaPipe
  pose/hand backends agree on landmark names via `perception/landmarks.py`
  (33-point BlazePose body topology, 21-point hand topology), so this
  module never needs to know which backend produced its input.
- `perception/artifact.py` writes a `<name>.parquet` feature table (via
  pandas/pyarrow) plus a `<name>.json` `PerceptionMetadata` sidecar
  (schema/preprocessing version, streaming SHA-256 video checksum, source
  vs. processed fps, coordinate convention, per-adapter model versions,
  tracker quality). Missing per-frame feature values are `NaN`, never a
  sentinel that could be confused with a real value.
- `perception/evaluation.py` computes detector precision/recall (bbox IoU
  or point-in-box), centre-point error (normalized and pixel), track
  coverage, longest missing interval, and interpolation rate against a
  lightweight `GroundTruthFrame` — `ground_truth_from_dataset_track`
  converts Prompt A's `dataset.schema.YoyoFrameAnnotation` rows into that
  shape without a hard import dependency in the other direction.
- `perception/overlay.py` renders a debug video (OpenCV) with the tracked
  bbox color-coded by confidence/interpolation state plus pose/hand
  keypoints drawn on top of the source frames.
- `yoyovision-perception` CLI (`perception/cli.py`) exposes `run` (write an
  artifact), `evaluate` (run + compare against a dataset record's ground
  truth, printing a JSON metrics report), and `overlay` (write a debug
  video) — see the README's "Running the perception pipeline standalone".

## Temporal event detector adapters (`ml/src/yoyovision_ml/events/`)

Prompt C's trainable temporal trick-event model. Three `TemporalEventDetector`
adapters are registered:

| Name | Registered as | `model_name` / `model_version` | Requires | Behavior when unconfigured/missing |
| --- | --- | --- | --- | --- |
| `MajorityClassEventDetector` | `"majority"` | `majority-class-baseline` / `1.0.0-not-a-trained-model` | `.fit(train_samples)` first (predicts one event per clip: the single most frequent `(class, outcome)` pair seen at fit time) | N/A — always available, never trained |
| `ThresholdRuleEventDetector` | `"rules"` | `threshold-rule-baseline` / `1.0.0-not-a-trained-model` | none (hand-crafted fixed thresholds over `yoyo_velocity`/`yoyo_direction_deg`/`hand_distance`) | N/A — always available, never trained |
| `PyTorchTemporalEventDetector` | `"torch"` | read from the checkpoint's `EventModelMetadata.model_name`/`model_version` (suffixed `+torch<version>`) | `pip install 'yoyovision-ml[torch]'` **and** a checkpoint written by `events.checkpoint.save_checkpoint` | Raises `ModelWeightsNotConfiguredError` if no `weights_path`/`YOYOVISION_TORCH_EVENT_WEIGHTS` is set or the path doesn't exist; raises `MissingOptionalDependencyError` if `torch` itself is absent — same two-stage fail-clear pattern as the perception adapters above, never a silent fallback |

`PyTorchTemporalEventDetector` always returns an empty `DeductionPrediction`
list: Prompt C's 20 classes deliberately exclude the 3 equipment-event
families in `domain.EventFamily` (broken string, tangled string, dropped
yo-yo), so equipment-deduction detection remains
`adapters_mock.MockTemporalEventDetector`'s job until a dedicated model
exists. Every prediction's `difficulty_band` is `UNKNOWN` — difficulty
banding is scoring-ruleset territory (see `docs/ruleset.md`), not something
the event model itself claims.

### Training, calibration, decoding, and evaluation

- `events/train.py`'s `train_model()` runs `windowing.py`'s
  normalization/windowing over a player-grouped split (`train.py`'s
  `player_grouped_split()`, with an explicit leakage assertion that no
  player id appears in more than one split), trains `model.py`'s dilated-
  conv1d TCN (lazily-imported `torch`, three heads: multi-label
  classification, start/end boundary, outcome) with class-balanced loss
  weighting and early stopping on validation loss, then fits
  `calibration.py`'s per-class temperature scaling on the validation split.
  `events/checkpoint.py` writes the result as a `.pt` weights file plus a
  `.json` `EventModelMetadata` sidecar (training config, feature names,
  normalization stats, calibration temperatures, player-split assignment,
  best epoch, val/test metrics, `torch` version, and
  `training_data_source` — always `"synthetic"` today).
- `events/decode.py` turns per-frame calibrated probabilities into discrete
  `EventDetection`s: run-length grouping per class, boundary-head-informed
  start/end refinement, a configurable choice of temporal NMS or event
  merging (`InferenceConfig.nms_strategy`), and uncertainty routing — a
  detection below `InferenceConfig.uncertainty_threshold` either becomes
  `unknown_technical_element` or gets `needs_review=True`, per Prompt C's
  "Add an uncertainty threshold that converts low-confidence predictions to
  unknown_technical_element or sends them to human review."
- `events/metrics.py`'s `evaluate()`/`evaluate_detector()` report event
  precision/recall, macro and per-class F1, temporal mAP at configurable
  tIoU thresholds, start/end boundary error, outcome-classification F1,
  calibration error (ECE + Brier), and a confusion matrix — the exact metric
  list Prompt C asks for.
- `yoyovision-events compare-baselines` runs `"majority"`, `"rules"`, and
  three trained ablations (`feature_subset="skeleton"` / `"trajectory"` /
  `"fused"`, i.e. skeleton-only, yo-yo-trajectory-only, and fused-feature
  models) against the *same* player-grouped split of one synthetic dataset,
  satisfying Prompt C's "Provide a comparison against simple baselines"
  requirement. Every code path that touches synthetic data prints/labels
  itself accordingly — see `events/cli.py`'s module docstring — so nothing
  from this comparison can be mistaken for a real-world accuracy claim.

See the README's "Temporal trick-event model" section for CLI usage and
"Current model status" for why none of this is wired into the default
pipeline yet.

## Scoring & judge calibration (`ml/src/yoyovision_ml/scoring/`)

Prompt D. Unlike every section above, this package doesn't add a new
`Protocol`/registry adapter — `ScoringEngine` stays `scoring_engine.
DeterministicScoringEngine` (see "Why Protocols, not concrete classes"
above and `docs/ruleset.md` for why scoring stays rules-based). Instead it
adds a layer of profile-aware input selection, human-input aggregation, and
offline evaluation *around* that same engine:

| Module | Adds |
| --- | --- |
| `profiles.py` | `ScoringProfile` (`practice`/`judge_assist`/`research`) and `ScoringProfileConfig` presets controlling which review statuses are eligible, which `RepeatedElementPolicyType` to use, whether automatic Freestyle Evaluation estimators run, whether multiple judges are required, and whether a bootstrap confidence interval is computed. |
| `overrides.py` | `apply_overrides` — per-event manual field edits with a full audit-log trail, applied to a copy of the event list. |
| `judges.py` | `aggregate_judge_scores` (multi-judge Freestyle Evaluation mean + disagreement warnings), `pairwise_judge_agreement`, and `match_clicks_to_events` (manually-entered judge timestamp clicks matched to detected events within a configurable tolerance). |
| `fe_estimators.py` | Optional heuristic estimators for 5 of the 8 manual Freestyle Evaluation categories (`execution`, `control`, `trick_diversity`, `space_use_emphasis`, `body_control`); `music_choreography`/`music_construction` never produce a numeric value (no audio analysis exists) and `showmanship` is never estimated at all. Every estimate reports a confidence, its supporting features, `model_name`/`model_version`, and an explicit "artistic scoring is subjective" warning — never presented as equivalent to a human judge's score. |
| `dangerous_play.py` | `detect_dangerous_play` — a hand-crafted, velocity-threshold heuristic that produces review-only `dangerous_play_review` `DeductionPrediction`s. Never auto-scored; see `docs/ruleset.md`'s "Dangerous-play review flags". |
| `pipeline.py` | `run_scoring_pipeline` — the orchestrator: overrides → profile/confirmation filtering → judge aggregation → automatic Freestyle Evaluation gap-filling → `DeterministicScoringEngine.calculate` → an optional bootstrap-resampled confidence interval, returned as one auditable `ScoringPipelineResult`. |
| `calibration.py` | Mean absolute error, Pearson and Spearman correlation, ICC(3,1), Bland–Altman summaries, event-count precision/recall, and an optional matplotlib calibration scatter plot (`pip install 'yoyovision-ml[plotting]'`) for comparing model output against expert judges. |
| `cli.py` | The `yoyovision-scoring` console script (`score` / `calibrate`) — see the README's "Scoring and judge calibration" section. |

Like `events/`'s baseline comparisons, `calibrate` draws no
production-readiness conclusions from a small sample — it's a tool for
comparing model output against whatever expert-judge ground truth is
supplied, not a claim about real-world accuracy. Kept `ml`-only for now:
no new `api` routes or DB tables persist judge clicks, overrides, or
Freestyle Evaluation estimates yet, and `api.services.scoring_service`
only wires in the one piece of this package needed for the API's existing
recompute flow — `scoring_engine.deduction_is_scorable`'s confirmation gate
— not profiles, judges, or estimators.

## Pipeline orchestration (`pipeline.py`)

`run_analysis_pipeline()` is the single function that resolves every
adapter by name and threads them together: preprocessing (frame
extraction, timestamp-preserving) → `PoseEstimator.predict` /
`HandEstimator.predict` → `YoyoDetector.predict` per frame batch →
`ObjectTracker.update` per timestamp → `StringAnalyzer.analyze` →
`FeatureExtractor.extract` → `TemporalEventDetector.predict` →
`ScoringEngine.calculate`. It is framework-free (no Celery/FastAPI import),
so the exact same call is used by `workers/tasks.py` in production and
directly in `ml`'s own pipeline tests — there is no separate "test mode"
pipeline that could drift from what actually runs.

## The model registry: caching and checksums on top of the adapter registry (Prompt F)

`inference/model_registry.py`'s `ModelRegistry` is a different concept from
`adapters_registry.py` above and deliberately doesn't replace it:

- `adapters_registry.py` answers "which *class* implements the
  `"pytorch"` yo-yo detector?" — a name→factory mapping, resolved fresh
  every call.
- `ModelRegistry` answers "have we already *constructed* the `"pytorch"`
  yo-yo detector in this process, and was its artefact's checksum verified
  before we did?" — `get_or_load(cache_key, loader, spec=...)` calls the
  supplied zero-arg `loader` (typically a closure around
  `create_yoyo_detector(...)` etc.) at most once per explicit `cache_key`
  for the life of the worker process, and `describe()` returns a read-only,
  path-free summary (name/version/device/load duration/checksum-verified
  flag per cache key) so `pipeline_runner.py`/`workers/health.py` can report
  exactly what's loaded without ever re-deriving or exposing a filesystem
  path.

`checksums.py`'s `verify_file_checksum(path, expected_sha256, label=...)`
runs once, before `get_or_load`'s `loader()` runs, whenever
`ModelArtifactSpec.path` is set — a mismatch (or a missing file) raises
`ModelIntegrityError` (a `DeterministicPipelineError`; see
`docs/architecture.md`'s "Production inference (Prompt F)") immediately,
never a silent load of a possibly-tampered-with or corrupted artefact. If
`expected_sha256` is `None` (no checksum pinned yet for that artefact),
verification is skipped but a warning is logged — this keeps local
development unblocked before a real checkpoint's checksum has been
recorded, while still making the gap visible. `ModelArtifactSpec.path` is
`None` for adapters with no on-disk weights (mock adapters, MediaPipe's
bundled models, the hand-written `"rules"`/`"majority"`/`"kalman"`
adapters), which skips checksum verification entirely for those.

`run_analysis_pipeline()`'s optional `model_registry` parameter accepts a
`ModelRegistry` instance; when omitted, the pipeline falls back to calling
`adapters_registry.create_*` directly, exactly as it did before Prompt F —
so nothing above is required to run the pipeline, only to run it
efficiently and verifiably at production scale.
