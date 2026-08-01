"""Add user.role and multi-judge entry tables.

Revision ID: 0007_multi_judge_entries
Revises: 0006_pipeline_adapter_config
Create Date: 2026-08-01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0007_multi_judge_entries"
down_revision: str | None = "0006_pipeline_adapter_config"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("role", sa.String(length=16), nullable=False, server_default="user"),
    )

    op.create_table(
        "judging_entries",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("ruleset_version", sa.String(length=32), nullable=False),
        sa.Column("ai_mix_profile", sa.String(length=8), nullable=False),
        sa.Column("aggregation_mode", sa.String(length=16), nullable=False),
        sa.Column("created_by", sa.String(length=36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_judging_entries_created_by", "judging_entries", ["created_by"])

    op.create_table(
        "judging_entry_videos",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "entry_id",
            sa.String(length=36),
            sa.ForeignKey("judging_entries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "video_id",
            sa.String(length=36),
            sa.ForeignKey("video_assets.id"),
            nullable=False,
        ),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column(
            "official_analysis_id",
            sa.String(length=36),
            sa.ForeignKey("analysis_jobs.id"),
            nullable=True,
        ),
        sa.Column(
            "shadow_analysis_id",
            sa.String(length=36),
            sa.ForeignKey("analysis_jobs.id"),
            nullable=True,
        ),
        sa.UniqueConstraint("entry_id", "video_id", name="uq_judging_entry_video"),
    )
    op.create_index("ix_judging_entry_videos_entry_id", "judging_entry_videos", ["entry_id"])

    op.create_table(
        "judge_assignments",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "entry_id",
            sa.String(length=36),
            sa.ForeignKey("judging_entries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("display_name", sa.String(length=128), nullable=False),
        sa.Column("invite_token_hash", sa.String(length=64), nullable=False),
        sa.Column("token_prefix", sa.String(length=16), nullable=False),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("include_in_results", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_shadow", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("invite_token_hash", name="uq_judge_invite_token_hash"),
    )
    op.create_index("ix_judge_assignments_entry_id", "judge_assignments", ["entry_id"])

    op.create_table(
        "judge_freestyle_scores",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "assignment_id",
            sa.String(length=36),
            sa.ForeignKey("judge_assignments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "entry_video_id",
            sa.String(length=36),
            sa.ForeignKey("judging_entry_videos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("execution", sa.Float(), nullable=True),
        sa.Column("control", sa.Float(), nullable=True),
        sa.Column("trick_diversity", sa.Float(), nullable=True),
        sa.Column("space_use_emphasis", sa.Float(), nullable=True),
        sa.Column("music_choreography", sa.Float(), nullable=True),
        sa.Column("music_construction", sa.Float(), nullable=True),
        sa.Column("body_control", sa.Float(), nullable=True),
        sa.Column("showmanship", sa.Float(), nullable=True),
        sa.Column("notes", sa.String(length=4096), nullable=False, server_default=""),
        sa.Column("is_submitted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "assignment_id", "entry_video_id", name="uq_judge_fe_assignment_video"
        ),
    )
    op.create_index(
        "ix_judge_freestyle_scores_assignment_id", "judge_freestyle_scores", ["assignment_id"]
    )


def downgrade() -> None:
    op.drop_table("judge_freestyle_scores")
    op.drop_table("judge_assignments")
    op.drop_table("judging_entry_videos")
    op.drop_table("judging_entries")
    op.drop_column("users", "role")
