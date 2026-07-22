# Data Model

Every domain object is defined once, as a plain, framework-free dataclass or
`StrEnum` in `ml/src/yoyovision_ml/domain.py`. The `api` package's
SQLAlchemy ORM models (`api/src/yoyovision_api/db_models.py`) and Pydantic
schemas, and the `workers` package's pipeline output, all reuse those same
enums directly — there is exactly one `EventFamily`, one `Outcome`, one
`ReviewStatus`, etc. in the whole codebase.

## Enums (`domain.py`)

| Enum | Values |
| --- | --- |
| `VideoStatus` | `uploaded`, `validating`, `ready`, `rejected`, `deleted` |
| `JobStatus` | `pending`, `running`, `completed`, `failed`, `cancelled` |
| `PipelineStage` | `queued`, `media_validation`, `preprocessing`, `pose_extraction`, `hand_extraction`, `yoyo_detection`, `tracking`, `string_analysis`, `feature_extraction`, `temporal_event_detection`, `scoring`, `done` — surfaced live as `AnalysisJob.current_stage` |
| `EventFamily` | `mount`, `hop`, `laceration`, `whip_catch`, `slack`, `suicide`, `rejection`, `roll`, `underpass`, `overpass`, `bind`, `return`, `regeneration`, `horizontal`, `fingerspin`, `body_trick`, `control_miss`, `landing_miss`, `catch_miss`, `yoyo_stop`, `yoyo_change`, `yoyo_detach`, `unknown_technical_element` |
| `Outcome` | `success`, `miss`, `uncertain` |
| `DifficultyBand` | `basic`, `intermediate`, `advanced`, `unknown` — **model/human-assigned, never an official trick value** (see `docs/ruleset.md`) |
| `Source` | `model`, `human`, `imported` |
| `ReviewStatus` | `pending`, `confirmed`, `rejected`, `edited` |
| `DeductionType` | `yoyo_stop`, `yoyo_change`, `yoyo_detach`, `dangerous_play_review`, `other` |

Three frozensets partition `EventFamily` for scoring purposes and are the
single source of truth for "does this family earn positive credit / count
as a mistake / trigger equipment deduction logic":
`POSITIVE_EVENT_FAMILIES`, `MISTAKE_EVENT_FAMILIES`,
`EQUIPMENT_EVENT_FAMILIES`.

## Core persisted models

Each table below lists `domain.py` dataclass fields alongside the
corresponding `db_models.py` ORM column (type/constraints). All primary keys
are server-generated UUID strings (`String(36)`, default `uuid.uuid4()`).

### `VideoAsset` / `video_assets`

| Field | Column type | Notes |
| --- | --- | --- |
| `id` | `String(36)` PK | |
| `owner_id` | `String(36)` FK → `users.id`, indexed | Ownership boundary for every query |
| `original_filename` | `String(512)` | Client-supplied, truncated to 512 chars, never used to build a path |
| `storage_key` | `String(512)`, unique | Server-generated; see Security below |
| `mime_type` | `String(128)` | Declared type, validated against sniffed signature at upload |
| `duration_ms` | `Integer`, nullable | From `ffprobe` |
| `width` / `height` | `Integer`, nullable | From `ffprobe` |
| `fps` | `Float`, nullable | From `ffprobe` |
| `file_size` | `Integer` | Bytes |
| `status` | `VideoStatus` enum column | |
| `created_at` | `DateTime(timezone=True)`, server default `now()` | |
| `deleted_at` | `DateTime(timezone=True)`, nullable | Soft-delete tombstone (see Deletion below) |

Relationship: `jobs: list[AnalysisJobORM]`, cascade `all, delete-orphan`.

### `AnalysisJob` / `analysis_jobs`

| Field | Column type | Notes |
| --- | --- | --- |
| `id` | `String(36)` PK | |
| `video_id` | `String(36)` FK → `video_assets.id`, indexed | |
| `status` | `JobStatus` enum column, default `pending` | |
| `progress` | `Float`, default `0.0` | `0.0`–`1.0` |
| `current_stage` | `PipelineStage` enum column, nullable | Live observability of where in the pipeline the job is |
| `error_code` / `error_message` | `String(64)` / `String(2048)`, nullable | Populated on `status=failed` |
| `pipeline_version` | `String(32)` | Stamped at job creation; lets a completed job be traced to the exact pipeline code version that produced it |
| `created_at` / `started_at` / `completed_at` | `DateTime(timezone=True)` | |

Relationships (all `cascade="all, delete-orphan"`, i.e. deleting a job
deletes everything derived from it): `events`, `deductions`,
`freestyle_evaluation` (one-to-one), `score_breakdown` (one-to-one).

### `AnalysisEvent` / `analysis_events`

| Field | Column type | Notes |
| --- | --- | --- |
| `id` | `String(36)` PK | |
| `analysis_id` | `String(36)` FK → `analysis_jobs.id`, indexed | |
| `label` | `String(128)` | Human-readable trick name, e.g. `"double-or-nothing"` |
| `family` | `EventFamily` enum column | |
| `start_ms` / `end_ms` | `Integer` | Bounded, atomic — original video timestamps, preserved end-to-end (product principle #9) |
| `confidence` | `Float` | `0.0`–`1.0`; drives the low-confidence review flag |
| `outcome` | `Outcome` enum column | |
| `difficulty_band` | `DifficultyBand` enum column | |
| `source` | `Source` enum column | `model` (detector output) / `human` (added via UI) / `imported` |
| `review_status` | `ReviewStatus` enum column, default `pending` | Set to `edited` on user edit, `confirmed`/`rejected` on explicit review action |
| `model_name` / `model_version` | `String(128)` / `String(64)`, nullable | Null for human-sourced events; required traceability for model-sourced ones (product principle #2) |
| `evidence_json` | `JSON`, default `{}` | Serialized `EvidenceRef` data — frame timestamp, optional bounding box, keypoint refs, note; rendered by the frontend's canvas overlay |
| `created_at` / `updated_at` | `DateTime(timezone=True)`, `onupdate=now()` | |

### `MajorDeduction` / `major_deductions`

| Field | Column type | Notes |
| --- | --- | --- |
| `id` | `String(36)` PK | |
| `analysis_id` | `String(36)` FK → `analysis_jobs.id`, indexed | |
| `type` | `DeductionType` enum column | |
| `timestamp_ms` | `Integer` | Original video timestamp |
| `quantity` | `Integer`, default `1` | Number of occurrences this row represents |
| `points` | `Float` | Points attributed (informational; the scoring engine recomputes from the ruleset, not by trusting this stored value, at recompute time) |
| `confidence` | `Float` | |
| `source` | `Source` enum column | |
| `review_status` | `ReviewStatus` enum column, default `pending` | For `dangerous_play_review` specifically (Prompt D), `pending` never contributes score impact — `scoring_engine.deduction_is_scorable` excludes any deduction type whose ruleset rule sets `requires_manual_confirmation=True` until a human sets this to `confirmed`. See `docs/ruleset.md`'s "Dangerous-play review flags". |

### `FreestyleEvaluation` / `freestyle_evaluations`

One row per `AnalysisJob` (`analysis_id` unique FK). All 8 category fields
(`execution`, `control`, `trick_diversity`, `space_use_emphasis`,
`music_choreography`, `music_construction`, `body_control`, `showmanship`)
are `Float | None` — nullable because they are manual-entry placeholders
until a human judge fills them in (product scope: "Freestyle Evaluation
placeholders and manual values"). Plus `source` (`Source` enum, default
`human`) and `notes` (`String(4096)`, default `""`).

### `ScoreBreakdown` / `score_breakdowns`

One row per `AnalysisJob` (`analysis_id` unique FK), overwritten on
recompute. Fields: `technical_raw`, `technical_scaled`,
`freestyle_evaluation_raw`, `freestyle_evaluation_scaled`,
`major_deductions`, `final_score`, `confidence` (all `Float`),
`ruleset_version` (`String(32)`), `warnings` (`JSON` list of strings — the
full audit trail described in `docs/ruleset.md`), `created_at`.

### `User` / `users`

`id` (PK), `email` (`String(255)`, unique, indexed), `hashed_password`
(`String(255)`, bcrypt via passlib), `created_at`. Dev-only JWT auth; no
OAuth/social login in MVP scope.

## Pipeline-intermediate types (not persisted)

`domain.py` also defines the in-memory types passed between pipeline
stages, never written to a table directly (only their *derived*
`AnalysisEvent`/`MajorDeduction` rows are persisted): `BoundingBox`,
`EvidenceRef`, `Keypoint`, `PoseFrame`/`PoseSequence`,
`HandFrame`/`HandSequence`, `Detection`, `Track`, `FeatureFrame`/`FeatureSet`,
and the pre-persistence prediction types `AnalysisEventPrediction` /
`DeductionPrediction` (what a `TemporalEventDetector` returns, before the
`workers` package turns each into a DB row with a generated `id` and
`review_status=pending`).

Prompt D's judge/override/calibration types (`JudgeClick`,
`JudgeFreestyleScore`, `EventOverride`, `FreestyleEvaluationEstimate`,
`ScoringPipelineResult`) live in `ml/src/yoyovision_ml/scoring/types.py`,
not `domain.py` — they are consumed by `scoring.pipeline.
run_scoring_pipeline` and the `yoyovision-scoring` CLI, but are not (yet)
persisted by any `api` ORM table or exposed by an `api` router; see
`docs/ruleset.md`'s "Scoring profiles, overrides, judges, and calibration"
section and `docs/adapters.md`'s "Scoring & judge calibration" section for
current scope.

## Ownership and privacy semantics

- Every query for a `VideoAsset`, `AnalysisJob`, event, deduction,
  evaluation, or score row filters by the requesting user's ownership chain
  (`VideoAssetORM.owner_id == current_user.id`, joined through
  `analysis_id` → `video_id` → `owner_id` for the child tables).
- A resource that exists but belongs to another user returns **404**, not
  403 — existence is never leaked to a non-owner (see `OwnedVideo` /
  equivalent dependency helpers in each router).
- Soft-deleted videos (`deleted_at IS NOT NULL`) are excluded from every
  list/get query (`VideoAssetORM.deleted_at.is_(None)` filters throughout
  `routers/videos.py`).

### Deletion (`DELETE /videos/{video_id}`)

Two modes, both always remove the stored video bytes from the active
`StoragePort` immediately:

- **Soft delete (default, `hard=false`)** — stamps `deleted_at`; the DB row
  (and its metadata) remains as an audit tombstone, but the video is
  inaccessible through any list/get/stream endpoint and its bytes are gone.
- **Hard delete (`hard=true`)** — additionally deletes the `VideoAssetORM`
  row itself; the `cascade="all, delete-orphan"` relationships mean every
  `AnalysisJob` and everything derived from it (events, deductions,
  evaluation, score) is deleted in the same transaction. This is the
  "permanent deletion of video and derived artefacts" required by the
  product spec.

## Frontend mirror

`frontend/src/lib/types.ts` hand-mirrors every enum and schema above (kept
in sync manually — see `docs/architecture.md`'s note on no shared codegen
yet). When adding or changing a field here, update, in order: `domain.py` →
`db_models.py` → the Pydantic schema in `api/src/yoyovision_api/schemas.py`
→ `frontend/src/lib/types.ts` → the relevant `frontend/src/hooks/*` and
components.
