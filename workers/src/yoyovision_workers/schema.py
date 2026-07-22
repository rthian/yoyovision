"""SQLAlchemy Core table definitions mirroring the tables owned by
`yoyovision_api.db_models` / `api/alembic/versions/0001_initial.py`.

The workers service intentionally does NOT import `yoyovision_api` (see
package docstring): it is a separately deployable process that shares the
Postgres *schema*, not Python code, with the API. Core `Table` objects (not a
full declarative ORM mapping) are used here because the worker only ever
performs simple inserts/updates on a handful of columns -- it never needs
relationships, lazy loading, or any of the richer ORM behaviour the API uses
for request/response mapping. If the API's migration changes these tables,
this file must be updated to match by hand; there is no shared source of
truth beyond the migration itself (documented in `docs/architecture.md`).
"""

from __future__ import annotations

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, MetaData, String, Table

metadata = MetaData()

video_assets = Table(
    "video_assets",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("owner_id", String(36), nullable=False),
    Column("original_filename", String(512), nullable=False),
    Column("storage_key", String(512), nullable=False),
    Column("mime_type", String(128), nullable=False),
    Column("duration_ms", Integer, nullable=True),
    Column("width", Integer, nullable=True),
    Column("height", Integer, nullable=True),
    Column("fps", Float, nullable=True),
    Column("file_size", Integer, nullable=False),
    Column("status", String(32), nullable=False),
    Column("created_at", DateTime(timezone=True)),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
)

analysis_jobs = Table(
    "analysis_jobs",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("video_id", String(36), nullable=False),
    Column("status", String(32), nullable=False),
    Column("progress", Float, nullable=False),
    Column("current_stage", String(32), nullable=True),
    Column("error_code", String(64), nullable=True),
    Column("error_message", String(2048), nullable=True),
    Column("pipeline_version", String(32), nullable=False),
    Column("created_at", DateTime(timezone=True)),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("completed_at", DateTime(timezone=True), nullable=True),
    #: Prompt F columns -- see `yoyovision_api.db_models.AnalysisJobORM` for
    #: the authoritative definitions/comments this mirrors.
    Column("model_versions", JSON, nullable=True),
    Column("device", String(32), nullable=True),
    Column("runtime_versions", JSON, nullable=True),
    Column("stage_durations_ms", JSON, nullable=True),
    #: `default=` (not just `nullable=False`) so existing `insert(...)` call
    #: sites -- including the test fixtures seeded before Prompt F -- keep
    #: working without passing these columns explicitly.
    Column("is_shadow", Boolean, nullable=False, default=False),
    Column("cancel_requested", Boolean, nullable=False, default=False),
    Column("retry_count", Integer, nullable=False, default=0),
)

analysis_events = Table(
    "analysis_events",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("analysis_id", String(36), nullable=False),
    Column("label", String(128), nullable=False),
    Column("family", String(48), nullable=False),
    Column("start_ms", Integer, nullable=False),
    Column("end_ms", Integer, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("outcome", String(16), nullable=False),
    Column("difficulty_band", String(16), nullable=False),
    Column("source", String(16), nullable=False),
    Column("review_status", String(16), nullable=False),
    Column("model_name", String(128), nullable=True),
    Column("model_version", String(64), nullable=True),
    Column("evidence_json", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True)),
    Column("updated_at", DateTime(timezone=True)),
)

major_deductions = Table(
    "major_deductions",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("analysis_id", String(36), nullable=False),
    Column("type", String(32), nullable=False),
    Column("timestamp_ms", Integer, nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("points", Float, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("source", String(16), nullable=False),
    Column("review_status", String(16), nullable=False),
)

score_breakdowns = Table(
    "score_breakdowns",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("analysis_id", String(36), nullable=False, unique=True),
    Column("technical_raw", Float, nullable=False),
    Column("technical_scaled", Float, nullable=False),
    Column("freestyle_evaluation_raw", Float, nullable=False),
    Column("freestyle_evaluation_scaled", Float, nullable=False),
    Column("major_deductions", Float, nullable=False),
    Column("final_score", Float, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("ruleset_version", String(32), nullable=False),
    Column("warnings", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True)),
)
