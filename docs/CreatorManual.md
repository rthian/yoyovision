# Creator Manual

A practical guide for creators, annotators, and builders who want YoYoVision
to detect 1A tricks more accurately and with higher confidence.

This manual explains **what the system does today**, **what is mock vs real**,
and **the step-by-step path** from uploaded footage to a trained trick-detection
model wired into the analysis pipeline.

For label-by-label annotation rules, see
[`annotation_handbook.md`](annotation_handbook.md). For swapping adapters in
code, see [`adapters.md`](adapters.md). For scoring disclaimers, see
[`ruleset.md`](ruleset.md).

---

## Who this is for

- **Creators** uploading routines and reviewing detections in the app
- **Annotators** building ground-truth datasets for training
- **ML builders** training perception and temporal event models

---

## Important context

**YoYoVision is a training and judge-assistance tool.** Scores and detections
are estimates until a human reviews them. They are not certified by IYYF, WYYC,
or any competition body.

**No trained trick-detection checkpoint ships with this repository today.**
The default analysis pipeline uses deterministic **mock** adapters for most
perception and event-detection stages. Re-running analysis on the same video
without changing adapters will not make detections “learn” your routine.

---

## What runs today

When you upload a video and run analysis, the worker executes this stack:

| Stage | Default adapter | What it does today |
| --- | --- | --- |
| Pose extraction | `mock` | Synthetic body landmarks (or `mediapipe` if configured) |
| Hand extraction | `mock` | Synthetic hand landmarks (or `mediapipe` if configured) |
| Yo-yo detection | `mock` | Fake bounding boxes per frame |
| Tracking | `mock` | Single pass-through track |
| String analysis | Rule-based | Deterministic string heuristics |
| Feature extraction | Rule-based | Merges pose/hand/track signals (~4 features in the worker path) |
| **Temporal event detection** | **`mock`** | **Synthetic trick timeline (~1 event / 2.5s, cycling families/outcomes)** |
| Scoring | Rules engine | Real, deterministic scoring from detected events |

### What that means in the review UI

If you see a mix of success / miss / uncertain events with moderate confidence
values, that pattern often reflects **mock temporal detection**, not a model
that watched your video and got confused.

Mock confidence is **not trainable** by re-running analysis. Replacing the mock
with a trained temporal model (and better upstream features) is what improves
detection quality.

### What already works without training

| Capability | Notes |
| --- | --- |
| Upload + analysis pipeline | End-to-end, mock-backed by default |
| Human review | Confirm, reject, edit, add, delete events |
| Live playhead scoring | Credits completed tricks; needs line-items API healthy |
| Routine window | Mark measure start and music stop; playback pauses at routine end |
| Freestyle Evaluation | Manual judge entry (required for freestyle score component) |
| Dataset format + CLI | Validate, split, import CVAT annotations |
| Temporal TCN training code | Runs today on **synthetic** features unless you wire real data |
| MediaPipe pose/hand | Real bundled models; swap adapter name to `mediapipe` |

---

## The detection stack (what you actually train)

Trick detection is a **pipeline**, not a single model:

```
Video
  → Pose + Hand estimation
  → Yo-yo detection per frame
  → Tracking (persistent yo-yo path)
  → Kinematic + string features
  → Temporal segmentation (TCN)
  → Trick events (start/end, family, label, outcome, confidence)
  → Deterministic scoring (rules engine — not learned)
```

### Impact order (highest first)

1. **Annotated trick events** — ground truth with `start_ms`, `end_ms`, `family`,
   `label`, `outcome`
2. **Temporal TCN** (`ml/src/yoyovision_ml/events/`) — learns boundaries, class,
   outcome, and confidence
3. **Yo-yo detector** — better trajectory → better features for the TCN
4. **Pose / hand** — MediaPipe works out of the box; domain fine-tuning is optional
5. **Tracker tuning** — Kalman tracker parameters (`max_gap_ms`, `static_camera`)

The **scoring engine is intentionally not learned.** Product principle: transparent
events + deterministic rules, not an opaque final-score model.

---

## Roadmap: from footage to accurate detection

### Phase 1 — Build a real dataset (biggest bottleneck)

Annotate real 1A routines into `DatasetRecord` JSON files. Schema:
`ml/src/yoyovision_ml/dataset/schema.py`. Annotator rules:
[`annotation_handbook.md`](annotation_handbook.md).

#### Minimum streams to start

| Stream | Purpose |
| --- | --- |
| `trick_events` | Time-bounded tricks: family, label, outcome, difficulty band |
| `yoyo_track` | Per-frame yo-yo position + visibility (trains/evaluates yo-yo detector) |

Optional later: `pose_landmarks`, `hand_landmarks`, `string_masks`, `judge_clicks`,
`freestyle_evaluations`.

#### Quality rules

- Use **milliseconds** from video start for all timestamps.
- `end_ms` must be strictly greater than `start_ms`.
- Prefer `outcome: uncertain` or `family: unknown_technical_element` over a wrong
  confident label. **Bad labels hurt training more than gaps.**
- Reuse stable **`player_id`** per performer so train/val/test splits do not leak
  the same player across sets (`ml/src/yoyovision_ml/dataset/splits.py`).

#### Dataset CLI

```bash
# Install ML package with CLI extras (from repo root, in your venv)
pip install -e "./ml[dev]"

# Validate all records in a dataset directory
yoyovision-dataset validate /path/to/dataset

# Summary statistics
yoyovision-dataset stats /path/to/dataset

# Player-grouped train / val / test split
yoyovision-dataset split /path/to/dataset --train 0.7 --val 0.15 --test 0.15

# Import CVAT exports (if you annotate in CVAT)
yoyovision-dataset import-cvat /path/to/cvat/export --output /path/to/dataset
```

Sample layout: `ml/sample_data/dataset_v1/` (synthetic placeholder — not real footage).

#### Using the review UI as a labeling source

The analysis review page supports confirm / reject / edit / add / delete on
events. That improves **scoring for judging today**. To use corrections for
**training**, export or convert reviewed events into `DatasetRecord.trick_events`
format. On the review page, **Add to training corpus** (visible after submit)
appends the adjudicated `DatasetRecord` plus video bytes into the corpus directory
configured by `DATASET_CORPUS_ROOT` (see `POST /analyses/{id}/export/corpus`).
You can still download a single-record JSON export for ad-hoc use.

---

### Phase 2 — Real perception (before or alongside event training)

#### Quick win: MediaPipe pose and hand

MediaPipe adapters are real and use bundled pretrained weights. Run the
standalone perception pipeline:

```bash
yoyovision-perception run video.mp4 --duration-ms 221860 --fps 60 \
  --pose-adapter mediapipe \
  --hand-adapter mediapipe \
  --tracker-adapter kalman
```

Evaluate against annotated `yoyo_track` in dataset records:

```bash
yoyovision-perception evaluate video.mp4 record.json \
  --duration-ms 221860 --fps 60 \
  --pose-adapter mediapipe --hand-adapter mediapipe
```

Overlay preview:

```bash
yoyovision-perception overlay video.mp4 --duration-ms 221860 --fps 60 \
  --output overlay.mp4 --pose-adapter mediapipe
```

**Note:** Celery workers read adapter names from env vars (`PIPELINE_POSE_ADAPTER`,
`PIPELINE_HAND_ADAPTER`, `PIPELINE_TRACKER_ADAPTER`, `PIPELINE_YOYO_ADAPTER`,
`PIPELINE_TEMPORAL_EVENT_ADAPTER`, plus checkpoint paths). See [`adapters.md`](adapters.md).

#### Yo-yo detector (train or supply weights)

No yo-yo detector checkpoint ships with this repo. Real slots exist:

- `pytorch` — set `YOYOVISION_TORCH_YOYO_WEIGHTS`
- `onnx` — set `YOYOVISION_ONNX_YOYO_MODEL`

Train against `yoyo_track` annotations:

```bash
yoyovision-perception train --dataset-dir /path/to/dataset --output-dir out --name yoyo \
  --sample-fps 15 --max-epochs 30
```

Then point the worker/env at your checkpoint (`PIPELINE_YOYO_ADAPTER=pytorch`,
`PIPELINE_YOYO_WEIGHTS=out/yoyo.pt`). Bad yo-yo detection degrades trajectory features
(`yoyo_velocity`, `yoyo_direction_deg`, etc.) that the temporal model depends on.

---

### Phase 3 — Train the temporal trick model (TCN)

The first **trainable** trick model in this repo lives under
`ml/src/yoyovision_ml/events/`. It is a temporal convolutional network (TCN) that
predicts:

- Trick **class** (20 atomic 1A families)
- **Start / end** boundaries
- **Outcome** (success / miss / uncertain)
- **Confidence** (calibrated when trained on real data)

Equipment events (`yoyo_stop`, `yoyo_change`, `yoyo_detach`) are **not** in the
TCN label space today; those remain separate.

#### Train (synthetic smoke test today)

```bash
yoyovision-events train --output-dir out --name model \
  --feature-subset fused --epochs 50
```

Checkpoints are written to `out/model.pt` with metadata including
`training_data_source` (today: `"synthetic"` unless you wire real data).

#### Run inference

```bash
yoyovision-events run --detector torch --weights out/model.pt \
  --feature-subset fused --output out/predictions.parquet
```

#### Evaluate

```bash
yoyovision-events evaluate --predictions out/predictions.parquet \
  --ground-truth /path/to/dataset

yoyovision-events compare-baselines --num-players 6 --clips-per-player 2
```

Set `YOYOVISION_TORCH_EVENT_WEIGHTS` (or pass weights via adapter kwargs) and use
`temporal_event_adapter_name="torch"` in `run_analysis_pipeline`.

#### Real-footage training path

`ml/scripts/prepare_training_corpus.py` chains:

1. `yoyovision-dataset validate` on a corpus directory
2. `PerceptionPipeline.run_and_write` for each adjudicated record
3. `yoyovision-events train --dataset-dir … --perception-dir …`

Populate the corpus from submitted reviews (`DATASET_CORPUS_ROOT` + **Add to
training corpus**), or assemble records manually under `videos/` and
`records/`.

#### Feature alignment (production)

`ml/src/yoyovision_ml/pipeline.py` now uses `compute_kinematic_features()`
(~18 kinematic columns) so worker inference matches TCN training inputs. Set
`PIPELINE_TEMPORAL_EVENT_ADAPTER=torch` and point weights at your checkpoint.

---

### Phase 4 — Wire trained models into production

1. **Configure adapters** — pose/hand `mediapipe`, yo-yo `pytorch`/`onnx` with
   weights, temporal `torch` with TCN checkpoint, tracker `kalman`.
2. **Align features** — worker kinematic features already match TCN training; verify checkpoint metadata
   at train and inference time.
3. **Per-job adapter profiles** — set `pipeline_adapter_config` when triggering a
   shadow analysis (`POST /videos/{id}/analyses`) or patch a pending job
   (`PATCH /analyses/{id}/pipeline-config`). Worker env vars remain the default.
4. **Re-run analyses** — compare events and confidence in the review UI; use
   shadow mode (`?shadow=true` on analysis create) to trial new weights without
   replacing the official result.
5. **Iterate** — evaluate on held-out **players**, fix labels, retrain.

See the four-step swap guide in [`adapters.md`](adapters.md):

1. Implement the `Protocol` in `ml/src/yoyovision_ml/interfaces.py`
2. Register in `ml/src/yoyovision_ml/adapters_registry.py`
3. Point configuration at the new adapter name + kwargs
4. Keep `"mock"` available for tests and demos

---

## What improves confidence specifically

| Lever | How it helps |
| --- | --- |
| **More labeled routines** | Diverse players, angles, lighting, music styles |
| **Consistent boundaries** | Agree when a trick starts/ends; see annotation handbook |
| **Better yo-yo track** | Cleaner trajectories → sharper temporal boundaries |
| **Player-grouped splits** | Prevents inflated confidence from memorizing one performer |
| **Honest uncertain labels** | Stops the model from learning false certainty |
| **Routine window** | Train and score only measure → music stop, not dead air |
| **Human review exports** | Confirmed events become high-quality training rows |
| **Calibration on val set** | Tune thresholds; do not tune on test footage |

Re-running mock analysis **does not** improve confidence. Replacing `mock-temporal-event-detector` with a trained `torch` detector does.

---

## Using the app effectively today (before custom training)

Even without a trained model, you can improve **judge-facing output**:

1. **Set routine window** — measure start and music stop so playback and scoring
   match the competitive span.
2. **Review every event** — confirm real tricks, reject hallucinations, edit
   boundaries and outcomes.
3. **Enter Freestyle Evaluation** — freestyle component stays 0 until a human
   fills the scorecard (by design).
4. **Re-run analysis in shadow mode** — test pipeline changes without overwriting
   the official job.
5. **Export** — JSON/CSV for offline review or dataset building.

After the line-items API is healthy, **live scoring** credits technical points
as tricks complete during playback (completed tricks only: `end_ms ≤ playhead`).

---

## Key file reference

| Topic | Path |
| --- | --- |
| Dataset schema | `ml/src/yoyovision_ml/dataset/schema.py` |
| Label ontology | `ml/src/yoyovision_ml/dataset/ontology/v1.yaml` |
| Annotation handbook | `docs/annotation_handbook.md` |
| Adapter swap guide | `docs/adapters.md` |
| Mock adapters | `ml/src/yoyovision_ml/adapters_mock.py` |
| Analysis pipeline | `ml/src/yoyovision_ml/pipeline.py` |
| Perception pipeline | `ml/src/yoyovision_ml/perception/pipeline.py` |
| Perception features | `ml/src/yoyovision_ml/perception/features.py` |
| Worker feature merge | `ml/src/yoyovision_ml/feature_extraction.py` |
| Temporal train loop | `ml/src/yoyovision_ml/events/train.py` |
| Temporal TCN model | `ml/src/yoyovision_ml/events/model.py` |
| Events CLI | `ml/src/yoyovision_ml/events/cli.py` |
| Worker runner | `workers/src/yoyovision_workers/pipeline_runner.py` |
| Worker settings | `workers/src/yoyovision_workers/config.py` |
| Sample dataset | `ml/sample_data/dataset_v1/` |

### Environment variables (real weights)

| Variable | Purpose |
| --- | --- |
| `YOYOVISION_TORCH_YOYO_WEIGHTS` | PyTorch yo-yo detector checkpoint |
| `YOYOVISION_ONNX_YOYO_MODEL` | ONNX yo-yo detector model |
| `YOYOVISION_TORCH_EVENT_WEIGHTS` | Temporal TCN checkpoint |

---

## Checklist: first real training run

Use this as a concrete order of operations for your first non-mock detector:

- [ ] Annotate **2–3 full 1A routines** (`trick_events` + `yoyo_track` minimum)
- [ ] Run `yoyovision-dataset validate` and `split` with player-grouped holds
- [ ] Run `yoyovision-perception` with `mediapipe` + `kalman` on those videos
- [x] Bridge dataset + perception Parquets → `ml/scripts/prepare_training_corpus.py`
- [ ] Evaluate on **held-out players**, not the train set
- [ ] Train or configure a **yo-yo detector** (`yoyovision-perception train --dataset-dir …`) if trajectory quality is weak
- [ ] Point worker at `temporal_event_adapter_name="torch"` + checkpoint path
- [ ] Align **feature extraction** between train and worker pipeline
- [ ] Re-run analysis (shadow first), review in UI, export corrections, retrain

---

## Related reading

- [`README.md`](../README.md) — project overview and “Current model status”
- [`annotation_handbook.md`](annotation_handbook.md) — how to label `DatasetRecord` files
- [`adapters.md`](adapters.md) — mock vs real adapters and swap procedure
- [`architecture.md`](architecture.md) — module map
- [`ruleset.md`](ruleset.md) — scoring rules and unofficial-draft disclaimer

---

## Summary

| Goal | What to do |
| --- | --- |
| Better tricks **right now** (no ML) | Review UI: confirm/reject/edit + routine window |
| Better tricks **with ML** | Annotate real footage → train TCN (+ yo-yo detector) → replace mock adapters |
| Higher **confidence** | More diverse labeled data, honest uncertain labels, player-grouped eval |
| Official **scores** | Human Freestyle Evaluation + reviewed events; scoring engine stays rule-based |

You cannot train detection from the upload UI alone today. The path is:
**annotated dataset → perception features → temporal model training → adapter swap in workers**.
