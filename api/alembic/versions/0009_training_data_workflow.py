"""Add video-source provenance and division-specific training annotations.

Revision ID: 0009_training_data_workflow
Revises: 0008_multi_division_foundation
Create Date: 2026-10-01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0009_training_data_workflow"
down_revision: str | None = "0008_multi_division_foundation"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "video_assets",
        sa.Column("source_type", sa.String(length=16), nullable=False, server_default="upload"),
    )
    op.add_column("video_assets", sa.Column("source_url", sa.String(length=2048)))
    op.add_column("video_assets", sa.Column("source_external_id", sa.String(length=128)))
    op.add_column("video_assets", sa.Column("player_id", sa.String(length=128)))
    op.add_column("video_assets", sa.Column("rights_confirmed_at", sa.DateTime(timezone=True)))

    op.create_table(
        "training_annotations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "video_id",
            sa.String(length=36),
            sa.ForeignKey("video_assets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(length=36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("division", sa.String(length=4), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=False),
        sa.Column("element_type", sa.String(length=64), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("technical_credit", sa.String(length=24), nullable=False),
        sa.Column("notes", sa.String(length=2048), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("start_ms >= 0", name="ck_training_annotations_start_nonnegative"),
        sa.CheckConstraint("end_ms > start_ms", name="ck_training_annotations_end_after_start"),
    )
    op.create_index(
        "ix_training_annotations_video_id", "training_annotations", ["video_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_training_annotations_video_id", table_name="training_annotations")
    op.drop_table("training_annotations")
    op.drop_column("video_assets", "rights_confirmed_at")
    op.drop_column("video_assets", "player_id")
    op.drop_column("video_assets", "source_external_id")
    op.drop_column("video_assets", "source_url")
    op.drop_column("video_assets", "source_type")
