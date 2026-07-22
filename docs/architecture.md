# Architecture

YoYoVision is four independently-testable Python/TypeScript packages plus
infrastructure, wired together by `docker-compose.yml`. This document is the
map; see `docs/data_model.md` for schemas, `docs/ruleset.md` for scoring, and
`docs/adapters.md` for how ML components get swapped.

## Package map

```
yoyovision/
├── ml/          yoyovision-ml       -- shared domain, interfaces, adapters, scoring
├── api/         yoyovision-api      -- FastAPI: persistence, auth, review, exports
├── workers/     yoyovision-workers  -- Celery: runs the ml pipeline, writes results
└── frontend/    (no package name)  -- Next.js: upload + review UI
```

`ml` has no dependency on `api` or `workers`. Both `api` and `workers`
depend on `ml` (for `yoyovision_ml.domain` enums/dataclasses, the scoring
engine, storage adapters, and the deterministic mock ML adapters) but never
on each other. The frontend talks to `api` only, over HTTP.

### `ml` — shared domain, interfaces, adapters, scoring

| Module | Responsibility |
| --- | --- |
| `domain.py` | Framework-agnostic enums (`EventFamily`, `Outcome`, `DifficultyBand`, `JobStatus`, …) and dataclasses (`VideoAsset`, `AnalysisEvent`, `ScoreBreakdown`, pipeline-intermediate types like `PoseSequence`/`Detection`/`Track`/`FeatureSet`). `api`'s ORM models and Pydantic schemas, and `workers`' pipeline, all import these enums directly so vocabulary never drifts. |
| `interfaces.py` | `Protocol`s for every replaceable component: `PoseEstimator`, `HandEstimator`, `YoyoDetector`, `ObjectTracker`, `StringAnalyzer`, `FeatureExtractor`, `TemporalEventDetector`, `ScoringEngine`, `StoragePort`. Calling code depends only on these, never on a concrete class. |
| `adapters_registry.py` | Name → factory registry (`register_pose_estimator("mock")`, `create_pose_estimator("mock")`, …). Swapping an adapter is a config string change, not an import change. See `docs/adapters.md`. |
| `adapters_mock.py` | The deterministic mock adapters that satisfy every `Protocol` above today (`MockPoseEstimator`, `MockHandEstimator`, `MockYoyoDetector`, `MockTracker`, `MockTemporalEventDetector`). Every `model_name` is prefixed `mock-`; output is a stable hash of the input, never a trained-model claim. |
| `media_validation.py` | MIME/signature sniffing + `ffprobe`-based container/stream metadata + quality checks, used by `api` at upload time. |
| `preprocessing.py` | Timestamp-preserving frame sampling (`extract_frames`); decodes real pixels via OpenCV when installed, degrades to timestamp-only `FrameRef`s otherwise (sufficient for the current mock detectors). |
| `string_analysis.py`, `feature_extraction.py` | Deterministic geometry/feature derivation from tracked positions, feeding the temporal event detector. |
| `pipeline.py` | `run_analysis_pipeline()` — the single, framework-free function that wires preprocessing → pose/hand/yoyo/tracking adapters → string/feature analysis → temporal event detection → scoring into one `PipelineResult`. Runs identically inside a Celery worker or a test. Prompt F added optional, backward-compatible `device_preference`/`model_registry`/`cancellation`/`stage_callback`/`reference_baseline` parameters and `stage_durations_ms`/`device`/`runtime_versions`/`monitoring` result fields — see `inference/` below and "Production inference (Prompt F)". |
| `inference/` | Prompt F (production inference) concerns, all optional and composed into `pipeline.py` rather than baked into it: `errors.py` (the `TransientPipelineError`/`DeterministicPipelineError` retry-classification taxonomy), `checksums.py` (SHA-256 model-artefact verification), `device.py` (`resolve_device` CPU/GPU selection with fallback, `runtime_versions()`), `timing.py` (`StageTimings`, per-stage duration recording), `cancellation.py` (`CancellationToken`, cooperative cancel/timeout polling between stages), `model_registry.py` (`ModelRegistry`, load-once-per-process adapter cache with checksum enforcement before load, path-free `describe()`), `monitoring.py` (class/confidence drift, failed-track rate), `report.py` (human-readable Markdown analysis report generation). See "Production inference (Prompt F)" below. |
| `ruleset.py` | `Ruleset` Pydantic model + YAML loader (`load_ruleset`, `default_ruleset`, `list_available_rulesets`). See `docs/ruleset.md`. |
| `scoring_engine.py` | `DeterministicScoringEngine`, the only implementation of the `ScoringEngine` protocol. Never predicts a score directly; only sums already-detected events/deductions under a `Ruleset`. Also exposes each stage (`technical_points`, `deduction_points`, `freestyle_evaluation_points`) and `deduction_is_scorable` (the dangerous-play confirmation gate) as standalone public functions. |
| `scoring/` | Prompt D — scoring profiles (`practice`/`judge_assist`/`research`), per-event manual overrides with an audit trail, multi-judge Freestyle Evaluation aggregation and click-to-event matching, optional heuristic Freestyle Evaluation estimators (never `showmanship`), a review-only dangerous-play flag detector, model-vs-judge calibration statistics (MAE/Pearson/Spearman/ICC/Bland-Altman/event-count precision-recall) plus an optional plot, the `run_scoring_pipeline` orchestrator, and the `yoyovision-scoring` CLI. Built on top of `scoring_engine.py`/`ruleset.py`, which remain the only place actual point math happens. Deliberately `ml`-only for now — see `docs/ruleset.md`'s "Scoring profiles, overrides, judges, and calibration" section and `docs/adapters.md`'s "Scoring & judge calibration" section. |
| `storage.py` | `LocalFilesystemStorage` and `S3CompatibleStorage`, both implementing `StoragePort`. Path-traversal-safe by construction (`_assert_safe_relative_key`). |
| `exports.py` | JSON report / CSV builders consumed by `api`'s export router. |
| `dataset/` | Versioned annotation-corpus schema, ontology, CVAT importer, split/stats/validation tooling, and the `yoyovision-dataset` CLI. See `docs/annotation_handbook.md`. |
| `perception/` | Standalone perception pipeline (real MediaPipe pose/hand adapters, PyTorch/ONNX yo-yo detector adapters, a hand-written Kalman tracker, kinematic feature computation, Parquet+JSON artifact I/O, evaluation metrics, debug overlay rendering) plus the `yoyovision-perception` CLI. Deliberately kept separate from `pipeline.py` — it stops at a feature artifact and never touches temporal event detection or scoring. See `docs/adapters.md`'s "Perception adapters" section. |
| `events/` | The trainable temporal trick-event model (Prompt C): 20-class label ontology (`labels.py`), Pydantic training/inference config (`config.py`), a numpy feature-windowing/normalization layer (`windowing.py`), a deterministic synthetic clip+event generator for development/testing (`synthetic.py`), a lazily-imported-`torch` dilated-conv1d TCN with classification/boundary/outcome heads (`model.py`), the player-grouped-split training loop with class-balanced loss, early stopping, and checkpointing (`train.py`, `checkpoint.py`), numpy-only per-class temperature-scaling calibration (`calibration.py`), numpy-only frame-probabilities→events decoding with configurable NMS/merge and uncertainty routing (`decode.py`), the shared inference pipeline tying normalization→model→calibration→decode→domain conversion together (`inference.py`, `convert.py`), a `TemporalEventDetector` adapter wrapping a trained checkpoint plus majority-class/threshold-rule baselines (`detector_torch.py`, `baselines.py`), evaluation metrics (`metrics.py`), Parquet+JSON prediction artifact I/O (`artifact.py`), and the `yoyovision-events` CLI (`cli.py`). Every checkpoint/prediction this package produces today is trained/evaluated on synthetic data only (no real annotated 1A footage exists yet) — see the README's "Current model status". |

### `api` — FastAPI service

Owns all persistence, auth, ownership enforcement, and every CRUD/export
HTTP endpoint. Routers (`api/src/yoyovision_api/routers/`):

| Router | Endpoints (summary) |
| --- | --- |
| `auth.py` | Dev-only JWT login (`POST /auth/login`) |
| `videos.py` | Upload, list, get, delete, trigger analysis (optionally `?shadow=true`, Prompt F), authenticated stream proxy |
| `analyses.py` | Get analysis job, cancel a running/queued job (Prompt F), get/recompute score |
| `events.py` | List/create/update/confirm/reject/delete `AnalysisEvent` rows |
| `deductions.py` | List/create/update/confirm/reject/delete `MajorDeduction` rows |
| `evaluations.py` | Get/upsert the manual `FreestyleEvaluation` |
| `exports.py` | JSON report, events CSV, deductions CSV |
| `rulesets.py` | List/get versioned `Ruleset`s (transparency) |

Cross-cutting modules: `db.py` (async SQLAlchemy engine/session), `config.py`
(`pydantic-settings` from env), `auth.py`/`security.py` (JWT, ownership
checks, upload validation glue, storage-key generation), `celery_client.py`
(enqueues analysis jobs onto the `workers` Celery app), `logging_setup.py`
(structured JSON logs with request/job/video IDs).

`main.py` also exposes `GET /health` (liveness -- process is up, never
touches Postgres) and `GET /health/ready` (readiness, Prompt F -- runs
`SELECT 1` through the same `get_db_session` dependency every other route
uses, returning `503` rather than raising on failure).

Every resource-scoped endpoint enforces ownership by filtering on
`owner_id`/the owning `analysis_id` chain and returning a plain 404 (never
403) for a resource that exists but isn't owned by the caller, so existence
of another user's data is never leaked.

### `workers` — Celery pipeline runner

| Module | Responsibility |
| --- | --- |
| `celery_app.py` | Declares the `cpu` and `gpu` `Queue`s and routes `run_analysis_pipeline_task` to `cpu` today (see docstring — no GPU-bound adapter exists yet, but the topology is ready for one). |
| `tasks.py` | The Celery task(s): `run_analysis_pipeline_task` (`bind=True`, `autoretry_for=(TransientPipelineError,)` with backoff, Prompt F) and `health_check_task` (Prompt F liveness/readiness probe). |
| `pipeline_runner.py` | Glue between the Celery task and the framework-free `ml.pipeline` call: fetches the video from storage, resolves the active `Ruleset`, runs the pipeline on a worker thread while cooperatively polling `cancel_requested` (Prompt F), persists events/deductions/score idempotently (delete-then-insert per job id), classifies failures via `inference.errors`, and writes `report.md`/`result.json` artefacts through storage. See "Production inference (Prompt F)". |
| `health.py` | `check_worker_health` (Prompt F) — Postgres reachability + the process-wide `ModelRegistry`'s loaded-model summary, wrapped by `tasks.health_check_task`. |
| `storage_factory.py` | Resolves the configured `StoragePort` (`STORAGE_BACKEND=local\|s3`) via `adapters_registry.create_storage_backend`. |
| `db.py`, `config.py`, `schema.py`, `logging_setup.py` | Async DB session, env settings, worker-local Pydantic schemas, structured logging — mirroring `api`'s equivalents so both packages log with the same job/video/model-version fields. `schema.py` must be updated by hand whenever an Alembic migration changes `analysis_jobs` (no shared source of truth beyond the migration itself — see that module's docstring). |

Two worker services in `docker-compose.yml` (`worker-cpu`, `worker-gpu`)
consume the same image with different `celery ... -Q <queue>` commands, per
the "separate CPU and GPU task queues" requirement.

### `frontend` — Next.js review UI

Talks to `api` exclusively over `NEXT_PUBLIC_API_BASE_URL`; never touches
Postgres, Redis, or storage directly. See `frontend/src/lib/types.ts` and
`frontend/src/lib/api-client.ts`, which mirror the API's Pydantic
schemas/enums by hand (no codegen step yet — keep both sides in sync
manually when either changes). Structure:

- `src/lib/` — types, typed API client, formatting helpers, auth-token storage.
- `src/hooks/` — TanStack Query hooks per resource (videos, analyses/score,
  events, deductions, evaluation, rulesets) plus the `AuthProvider`.
- `src/components/` — upload form, video list, analysis job list, event
  timeline + table, deduction table, freestyle evaluation form, score
  breakdown panel, ruleset transparency panel, export buttons, video player
  with a canvas overlay that draws each event's evidence bounding box.
  `AnalysisJobList` also surfaces Prompt F job state: a "Cancel" action for
  `pending`/`running` jobs (`POST /analyses/{id}/cancel`), a "Cancelling..."
  label while `cancel_requested` is set, and a "Shadow" badge for
  `is_shadow` jobs.
- `src/app/` — `/` (upload + video list), `/videos/[videoId]` (analysis
  runs for one video, with a shadow-mode checkbox next to "Run analysis"
  that passes `?shadow=true`, Prompt F), `/analyses/[analysisId]` (the full
  review page: video+overlay, timeline, event/deduction editors, evaluation
  form, score, ruleset, exports; shows a non-official banner when
  `is_shadow` is set), `/login`.

## Request flow: upload → review

1. **Upload** — `POST /videos` (multipart). `api` sniffs MIME/signature,
   runs `ffprobe` for duration/width/height/fps, enforces size/duration
   limits, generates a server-side storage key (never derived from the
   client filename), and writes the `VideoAsset` row with `status=ready`.
2. **Trigger analysis** — `POST /videos/{id}/analyses` (optionally
   `?shadow=true`, Prompt F) creates an `AnalysisJob` row (`status=pending`)
   and enqueues a Celery task on the `cpu` queue via `celery_client.py`.
3. **Pipeline run** — a `worker-cpu` process picks up the task, fetches the
   video bytes from the configured `StoragePort`, calls
   `run_analysis_pipeline()` on a worker thread while cooperatively polling
   for cancellation, and persists the resulting `AnalysisEvent`,
   `MajorDeduction`, and `ScoreBreakdown` rows (idempotently — see below),
   updating `AnalysisJob.current_stage`/`progress` as it goes (see
   `domain.PipelineStage` for the full stage list) and `status=completed`
   (or `failed`/`cancelled`, with `error_code`/`error_message`) at the end.
   A human can request cancellation at any point via
   `POST /analyses/{id}/cancel` (Prompt F).
4. **Review** — the frontend polls `GET /videos/{id}/analyses` and
   `GET /analyses/{id}` while a job is pending/running, then renders the
   full review page once `completed`: video playback with an evidence
   overlay, the event/deduction tables (add/edit/delete/confirm/reject —
   product principle #4), a manual Freestyle Evaluation form, the
   `ScoreBreakdown` (with a "Recalculate" action hitting
   `POST /analyses/{id}/score/recompute` after edits), and the ruleset
   transparency panel.
5. **Export** — `GET /analyses/{id}/export/report.json`,
   `.../export/events.csv`, `.../export/deductions.csv`, all
   ownership-checked and filename-sanitized.

## Production inference (Prompt F)

Prompt F hardens `run_analysis_pipeline()` for unattended, at-scale use
without changing its default (mock-adapter, single-video, synchronous)
behavior — every addition below is an optional parameter or an additive
result field. All of it lives in `ml/src/yoyovision_ml/inference/`, `pipeline.py`
wires it in, and `workers/pipeline_runner.py` is the only caller that turns
these primitives into DB rows.

- **Model registry & checksums** (`inference/model_registry.py`,
  `checksums.py`) — `ModelRegistry` loads each configured model adapter at
  most once per worker process, verifies its artefact's SHA-256 against a
  pinned manifest *before* constructing the adapter, and exposes
  `describe()` (name/version/checksum, no filesystem paths) for
  persistence and reporting. A checksum mismatch is a
  `DeterministicPipelineError` — retrying won't fix a corrupt artefact.
- **Device resolution** (`inference/device.py`) — `resolve_device()` honors
  an explicit `device_preference` ("cpu"/"cuda"/"auto"), falls back to CPU
  when CUDA is requested but unavailable, and `runtime_versions()` captures
  the Python/PyTorch/CUDA versions actually used for a given run.
- **Batching** — `perception/detector_pytorch.py`'s yo-yo detector batches
  frames up to a bounded batch size instead of one-frame-at-a-time
  inference, transparent to callers.
- **Cooperative cancellation & timeouts** (`inference/cancellation.py`) — a
  thread-safe `CancellationToken` is polled at pipeline stage boundaries
  (not mid-stage); `pipeline_runner.py` runs the pipeline on a worker
  thread and flips the token when `AnalysisJob.cancel_requested` is set
  (via `POST /analyses/{id}/cancel`) or a deadline elapses, producing a
  distinct `cancelled` terminal status rather than `failed`.
- **Per-stage timing** (`inference/timing.py`) — `StageTimings` context
  manager records wall-clock duration per `PipelineStage`; surfaced as
  `PipelineResult.stage_durations_ms` and persisted to
  `AnalysisJob.stage_durations_ms`.
- **Monitoring** (`inference/monitoring.py`) — given an optional
  `reference_baseline`, computes class-distribution drift, confidence
  drift, and failed-track rate for the current run; attached to
  `PipelineResult.monitoring` and folded into the generated report.
- **Reporting** (`inference/report.py`) — renders a human-readable Markdown
  summary (model versions, device, timings, monitoring signals, event/score
  summary) that `pipeline_runner.py` writes to storage as `report.md`
  alongside a machine-readable `result.json`, best-effort (a storage
  failure here never fails the job).
- **Error classification & retries** (`inference/errors.py`) —
  `TransientPipelineError` (I/O timeouts, transient storage/DB errors —
  Celery should retry) vs `DeterministicPipelineError` (checksum mismatch,
  malformed video, adapter bugs — retrying is pointless). `tasks.py`'s
  `run_analysis_pipeline_task` is `bind=True, autoretry_for=(TransientPipelineError,)`
  with backoff; `AnalysisJob.retry_count` is incremented on each retry and
  the job is only marked `failed` once retries are exhausted or the error
  is deterministic.
- **Idempotent persistence** — because a retried task re-runs the whole
  pipeline, `pipeline_runner.py` deletes any `AnalysisEvent`/
  `MajorDeduction`/`ScoreBreakdown` rows already written for that
  `AnalysisJob.id` immediately before inserting the fresh set, so a retry
  never doubles up rows.
- **Shadow mode** — jobs created with `is_shadow=True` (`POST
  /videos/{id}/analyses?shadow=true`) run the identical pipeline and are
  persisted identically, but `is_shadow` lets the frontend/API hide them
  from "official" score views — useful for A/B-testing a new model or
  ruleset against production traffic without affecting what athletes see.
- **Health/readiness** — `api`'s `GET /health` (liveness) and `GET
  /health/ready` (Postgres reachability) plus `workers`' `health_check_task`
  (Postgres reachability + `ModelRegistry` summary, see `workers/health.py`)
  give orchestrators (Compose healthchecks, k8s probes) a real signal
  instead of "the process didn't crash yet."
- **Regression fixtures** — `ml/tests/inference/test_regression_fixtures.py`
  runs the full pipeline against the committed synthetic sample dataset
  (mock adapters) and asserts stable event counts/scores, guarding against
  silent behavior drift in the orchestration code across the changes above.

None of this requires GPU hardware or real model weights to exercise: every
test above runs against the mock adapters and a locally-generated synthetic
MP4, matching the rest of the repo's "runs fully offline, no license/GPU
needed" posture.

## Infrastructure (`docker-compose.yml`)

| Service | Image/build | Purpose |
| --- | --- | --- |
| `postgres` | `postgres:16-alpine` | Primary datastore |
| `redis` | `redis:7-alpine` | Celery broker (`/0`) + result backend (`/1`) |
| `minio` | `minio/minio` | S3-compatible storage for local parity testing (`STORAGE_BACKEND=s3`) |
| `api` | `api/Dockerfile`, context `.` | FastAPI app; runs `alembic upgrade head` then `uvicorn --reload` |
| `worker-cpu` / `worker-gpu` | `workers/Dockerfile`, context `.` | Celery workers on the `cpu` / `gpu` queues |
| `frontend` | `frontend/Dockerfile`, context `./frontend` | Next.js dev server |

`api` and `workers` both build from repo-root context because they need to
`COPY` the sibling `ml/` package into the image alongside their own source
(see each `Dockerfile`'s comments); `frontend` builds from its own directory
since it has no cross-package Python dependency.

## Known dependency constraints

`passlib[bcrypt]` (used for password hashing in `api/security.py`) is
unmaintained and probes `bcrypt.__about__`, which was removed in
`bcrypt>=4.1`. `api/pyproject.toml` therefore pins `bcrypt>=4.0,<4.1`; do not
remove that pin without re-verifying passlib compatibility.
