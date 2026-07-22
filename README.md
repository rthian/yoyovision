# YoYoVision

AI-assisted yo-yo freestyle analysis platform. Users upload a 1A freestyle video;
the system extracts pose/hand/yo-yo tracking, detects atomic trick events on a
timeline, applies a versioned deterministic scoring ruleset, and produces a
reviewable, editable scoring report.

## Important product disclaimer

**YoYoVision is a training and judge-assistance tool.** Scores produced by this
application are **not** officially certified by IYYF, WYYC, or any competition
body. All model-generated events, difficulty bands, and scores are estimates
that a human must review, edit, and confirm before being treated as
authoritative. See [`docs/ruleset.md`](docs/ruleset.md) for details.

## Architecture at a glance

- **`frontend/`** — Next.js (App Router, TypeScript strict) review UI: upload,
  video playback with pose/hand/yo-yo overlays, event timeline editor, score
  breakdown, JSON/CSV export.
- **`api/`** — FastAPI + SQLAlchemy 2 + Alembic + PostgreSQL. Owns persistence,
  auth, ownership checks, file validation, and all CRUD/export endpoints.
- **`workers/`** — Celery workers with separate `cpu` and `gpu` queues. Runs the
  offline analysis pipeline: ingestion → preprocessing → pose/hand extraction →
  yo-yo detection/tracking → feature extraction → temporal event detection →
  scoring.
- **`ml/`** — Shared, framework-agnostic domain models, `Protocol` interfaces
  for every ML component, adapter implementations (mock now, pluggable real
  models later), the deterministic scoring engine, and the versioned ruleset
  configuration.

See [`docs/architecture.md`](docs/architecture.md) for the full module map,
[`docs/data_model.md`](docs/data_model.md) for every persisted schema and
ownership/deletion semantics, and [`docs/adapters.md`](docs/adapters.md) for
how to swap in real model weights.

## Dataset and annotation system

`ml/src/yoyovision_ml/dataset/` defines a versioned, reproducible dataset
format for training the eventual trick-detection model: Pydantic schemas,
a versioned trick-label ontology (`dataset/ontology/v1.yaml`), player-grouped
split generation, cross-video validation, dataset statistics/agreement
reporting, a CVAT box/point-track importer, and a `yoyovision-dataset` CLI
(`validate` / `stats` / `split` / `import-cvat`). A synthetic sample dataset
lives under `ml/sample_data/dataset_v1/` (placeholder videos, hand-authored
annotations — not real footage). See
[`docs/annotation_handbook.md`](docs/annotation_handbook.md) for the
annotator-facing guide. No training model exists yet; this is dataset
infrastructure only.

## Perception pipeline (pose, hand, yo-yo detection/tracking)

`ml/src/yoyovision_ml/perception/` is a separate, standalone pipeline (kept
distinct from the scoring-oriented `ml/src/yoyovision_ml/pipeline.py`) that
runs real pose/hand estimation, yo-yo detection, and tracking, then computes
a per-frame kinematic feature table and writes it as a Parquet artifact plus
a JSON metadata sidecar (checksum, model versions, coordinate convention).
It ships:

- Real **MediaPipe** pose/hand adapters (`"mediapipe"`) — genuine inference
  the moment `pip install 'yoyovision-ml[mediapipe]'` is run, no separate
  checkpoint needed.
- Real **PyTorch**/**ONNX Runtime** yo-yo detector adapters (`"pytorch"` /
  `"onnx"`) — load whatever checkpoint/model file is explicitly configured
  and refuse to run un-configured (no trained weights ship with this repo;
  see "Current model status" below).
- A real, deterministic **Kalman-filter** tracker (`"kalman"`) with
  short-gap interpolation and a track-quality score.
- Evaluation metrics (detector precision/recall, centre-point error, track
  coverage, longest missing interval, interpolation rate) and a debug
  overlay-video renderer.
- The `yoyovision-perception` CLI (`run` / `evaluate` / `overlay`).

See [`docs/adapters.md`](docs/adapters.md) for the full adapter list and how
to add another one.

## Temporal trick-event model (training, inference, evaluation)

`ml/src/yoyovision_ml/events/` is the first trainable model in this
repository: a modest temporal convolutional network (TCN) that predicts
time-bounded 1A atomic trick events (20 classes — mount, hop, laceration,
whip_catch, slack, suicide, rejection, roll, underpass, overpass, bind,
return, regeneration, horizontal, fingerspin, body_trick, control_miss,
landing_miss, catch_miss, unknown_technical_element) with start/end
boundaries, an outcome (success/miss/uncertain), and a calibrated
confidence, from Prompt B's per-frame kinematic features. It ships:

- A small dilated-conv1d encoder plus three heads (multi-label
  classification, start/end boundary, outcome) — deliberately modest and
  reproducible, not a large video transformer.
- Player-grouped train/val/test splits with an explicit leakage check,
  class-balanced loss weighting, a configurable temporal window, a
  deterministic seed, early stopping, and full experiment configuration +
  checkpoint metadata saved with every run (`events/checkpoint.py`).
- Post-hoc per-class temperature-scaling calibration, configurable temporal
  NMS/merge, and an uncertainty threshold that relabels low-confidence
  predictions as `unknown_technical_element` or flags them for human review.
- Metrics: event precision/recall, macro/per-class F1, temporal mAP at
  configurable tIoU thresholds, start/end boundary error, outcome F1,
  calibration error (ECE + Brier), and a confusion matrix
  (`events/metrics.py`).
- Two always-available reference baselines (majority class, hand-crafted
  threshold rules) plus skeleton-only / trajectory-only / fused feature
  ablations of the trained model, all comparable on the same split via
  `compare-baselines`.
- A deterministic **synthetic** clip+event generator (`events/synthetic.py`)
  so the full pipeline is trainable and testable today, since no real
  annotated 1A footage exists yet (see "Current model status" below).
- The `yoyovision-events` CLI (`train` / `run` / `evaluate` /
  `compare-baselines`).

```bash
cd ml && pip install -e ".[dev,torch]"
yoyovision-events train --output-dir out --name model \
  --num-players 6 --clips-per-player 2 --num-events-per-clip 10
yoyovision-events run --detector torch --weights out/model.pt \
  --synthetic-seed 1 --synthetic-video-id demo --synthetic-player-id p1 \
  --output-dir out --name predictions
yoyovision-events evaluate --predictions out/predictions.parquet \
  --synthetic-seed 1 --synthetic-video-id demo --synthetic-player-id p1
yoyovision-events compare-baselines --num-players 6 --clips-per-player 2
```

See [`docs/adapters.md`](docs/adapters.md)'s "Temporal event detector
adapters" section for the full adapter list.

## Scoring and judge calibration

`ml/src/yoyovision_ml/scoring/` (Prompt D) builds on the deterministic
`scoring_engine.py`/`ruleset.py` (see [`docs/ruleset.md`](docs/ruleset.md))
with everything needed to run and evaluate scoring in different contexts:

- Named **scoring profiles** — `practice`, `judge_assist` (default), and
  `research` — controlling review-status gating, repeated-element policy,
  automatic Freestyle Evaluation estimators, and confidence intervals.
- **Per-event manual overrides** with a full audit-log trail, and support
  for **manually entered judge clicks** and **multiple human judge
  scores**, aggregated with agreement statistics.
- Optional heuristic **Freestyle Evaluation estimators** for 5 of the 8
  categories (never `showmanship`), each reporting a confidence,
  supporting features, and an explicit "artistic scoring is subjective"
  warning.
- A review-only **dangerous-play flag detector** — per Prompt D, this can
  never automatically disqualify a player; a flag always requires human
  confirmation before it affects a score (see `docs/ruleset.md`'s
  "Dangerous-play review flags").
- **Calibration** metrics for comparing model output against expert
  judges: mean absolute error, Pearson and Spearman correlation, ICC(3,1),
  Bland–Altman summaries, event-count precision/recall, and an optional
  calibration scatter plot.
- The `yoyovision-scoring` CLI (`score` / `calibrate`):

```bash
cd ml && pip install -e ".[dev,plotting]"
yoyovision-scoring score \
  --record sample_data/dataset_v1/records/sample_video_001__adjudicated.json \
  --profile judge_assist --bootstrap-iterations 500
yoyovision-scoring calibrate \
  --pair sample_data/dataset_v1/records/sample_video_001__adjudicated.json out/predictions.parquet \
  --plot-output out/calibration.png
```

Kept deliberately `ml`-only for now (no new API routes, DB tables, or
frontend UI) — see [`docs/adapters.md`](docs/adapters.md)'s "Scoring &
judge calibration" section for current scope and what is intentionally
deferred.

## Production inference

`ml/src/yoyovision_ml/inference/` and `workers/pipeline_runner.py` (Prompt F)
harden the analysis pipeline for unattended, at-scale operation, on top of
(never instead of) everything above:

- Model artefact **checksum verification** before load, and a process-wide
  **model registry** so a long-lived worker loads each configured model at
  most once, not once per job.
- **CPU/GPU device resolution** with automatic fallback, and bounded
  **batched inference** for the yo-yo detector.
- Cooperative **cancellation** (`POST /analyses/{id}/cancel`) polled at
  stage boundaries, per-stage **timing**, and **shadow-mode** runs
  (`POST /videos/{id}/analyses?shadow=true`) that execute the full pipeline
  without affecting official scores.
- **Monitoring signals** (class/confidence drift, failed-track rate) and a
  generated Markdown **report** per job.
- Transient-vs-deterministic **error classification** driving automatic
  Celery retries, **idempotent** DB persistence across retries, and
  **health**/**readiness** endpoints (`GET /health`, `GET /health/ready` on
  `api`; a Celery health-check task on `workers`).
- **Regression fixtures** (`ml/tests/inference/test_regression_fixtures.py`)
  that run the full pipeline against the committed synthetic dataset and
  guard against silent output drift.

See [`docs/architecture.md`](docs/architecture.md)'s "Production inference
(Prompt F)" for the full breakdown and
[`docs/adapters.md`](docs/adapters.md)'s "The model registry" for how it
relates to the adapter registry above.

## Current model status

**No trained model checkpoint ships with this repository, for either the
yo-yo detector or the temporal trick-event model.** The default pipeline
(`workers`/`api`) still runs the temporal-event-detection stage through a
deterministic, clearly labelled mock adapter (`model_name` starts with
`mock-`; see `ml/src/yoyovision_ml/adapters_mock.py`) so the rest of the
pipeline, scoring engine, and review UI can be fully exercised end-to-end.
The trainable temporal trick-event model above (`events/`) is real and
runnable, but every run produced by this repository today trains on
**synthetic** data only (`training_data_source: "synthetic"` in every
checkpoint's metadata) — it has not seen real annotated 1A footage, so its
metrics are a pipeline smoke test, not a measurement of real-world
accuracy, and it must not be read as production-ready (Prompt C: "Do not
claim production readiness based only on clip-level accuracy"). Pose/hand
estimation has real, working MediaPipe adapters (bundled pretrained
models, no extra checkpoint required) as of the perception pipeline above.
Nothing in this repository fabricates accuracy numbers or claims trained
weights that do not exist. See [`docs/adapters.md`](docs/adapters.md) for
how a real adapter gets swapped in.

## Local development

```bash
cp .env.example .env
docker compose up --build
```

- API: http://localhost:8000 (OpenAPI docs at `/docs`)
- Frontend: http://localhost:3000
- MinIO console: http://localhost:9001

### Running the perception pipeline standalone

```bash
cd ml && pip install -e ".[dev,mediapipe]"  # add ",torch" / ",onnx" for those detectors
yoyovision-perception run video.mp4 --duration-ms 20000 --fps 30 \
  --pose-adapter mediapipe --hand-adapter mediapipe \
  --tracker-adapter kalman --output-dir out --name clip
yoyovision-perception evaluate video.mp4 record.json --duration-ms 20000 --fps 30
yoyovision-perception overlay video.mp4 --duration-ms 20000 --fps 30 --output overlay.mp4
```

### Running tests

```bash
# Python (ml, api, workers)
cd ml && pip install -e ".[dev]" && pytest && ruff check . && mypy -p yoyovision_ml
cd api && pip install -e ".[dev]" && pytest && ruff check . && mypy -p yoyovision_api
cd workers && pip install -e ".[dev]" && pytest && ruff check . && mypy -p yoyovision_workers

# Frontend
cd frontend && npm install && npm run lint && npm run typecheck && npm run test
npx playwright install --with-deps && npm run test:e2e
```

## Repository layout

```
yoyovision/
├── api/        FastAPI service (persistence, auth, CRUD, exports)
├── workers/    Celery workers (analysis pipeline orchestration)
├── ml/         Shared domain models, ML interfaces, adapters, scoring engine
├── frontend/   Next.js review UI
├── docs/       Architecture, ruleset, data model, adapter-swap docs,
│               annotation handbook
└── scripts/    Dev bootstrap helpers
```
