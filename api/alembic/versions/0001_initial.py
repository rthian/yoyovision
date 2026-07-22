"""Initial schema: users, video_assets, analysis_jobs, analysis_events,
major_deductions, freestyle_evaluations, score_breakdowns.

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-19

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "video_assets",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("owner_id", sa.String(length=36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="uploaded"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_video_assets_owner_id", "video_assets", ["owner_id"])
    op.create_index("ix_video_assets_storage_key", "video_assets", ["storage_key"], unique=True)

    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "video_id", sa.String(length=36), sa.ForeignKey("video_assets.id"), nullable=False
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("current_stage", sa.String(length=32), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.String(length=2048), nullable=True),
        sa.Column("pipeline_version", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_analysis_jobs_video_id", "analysis_jobs", ["video_id"])

    op.create_table(
        "analysis_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "analysis_id", sa.String(length=36), sa.ForeignKey("analysis_jobs.id"), nullable=False
        ),
        sa.Column("label", sa.String(length=128), nullable=False),
        sa.Column("family", sa.String(length=48), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("difficulty_band", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("review_status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("model_name", sa.String(length=128), nullable=True),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )
    op.create_index("ix_analysis_events_analysis_id", "analysis_events", ["analysis_id"])

    op.create_table(
        "major_deductions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "analysis_id", sa.String(length=36), sa.ForeignKey("analysis_jobs.id"), nullable=False
        ),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("timestamp_ms", sa.Integer(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("points", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("review_status", sa.String(length=16), nullable=False, server_default="pending"),
    )
    op.create_index("ix_major_deductions_analysis_id", "major_deductions", ["analysis_id"])

    op.create_table(
        "freestyle_evaluations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "analysis_id", sa.String(length=36), sa.ForeignKey("analysis_jobs.id"), nullable=False
        ),
        sa.Column("execution", sa.Float(), nullable=True),
        sa.Column("control", sa.Float(), nullable=True),
        sa.Column("trick_diversity", sa.Float(), nullable=True),
        sa.Column("space_use_emphasis", sa.Float(), nullable=True),
        sa.Column("music_choreography", sa.Float(), nullable=True),
        sa.Column("music_construction", sa.Float(), nullable=True),
        sa.Column("body_control", sa.Float(), nullable=True),
        sa.Column("showmanship", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False, server_default="human"),
        sa.Column("notes", sa.String(length=4096), nullable=False, server_default=""),
    )
    op.create_unique_constraint(
        "uq_freestyle_evaluations_analysis_id", "freestyle_evaluations", ["analysis_id"]
    )

    op.create_table(
        "score_breakdowns",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "analysis_id", sa.String(length=36), sa.ForeignKey("analysis_jobs.id"), nullable=False
        ),
        sa.Column("technical_raw", sa.Float(), nullable=False),
        sa.Column("technical_scaled", sa.Float(), nullable=False),
        sa.Column("freestyle_evaluation_raw", sa.Float(), nullable=False),
        sa.Column("freestyle_evaluation_scaled", sa.Float(), nullable=False),
        sa.Column("major_deductions", sa.Float(), nullable=False),
        sa.Column("final_score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("ruleset_version", sa.String(length=32), nullable=False),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_unique_constraint(
        "uq_score_breakdowns_analysis_id", "score_breakdowns", ["analysis_id"]
    )


def downgrade() -> None:
    op.drop_table("score_breakdowns")
    op.drop_table("freestyle_evaluations")
    op.drop_table("major_deductions")
    op.drop_table("analysis_events")
    op.drop_table("analysis_jobs")
    op.drop_table("video_assets")
    op.drop_table("users")
