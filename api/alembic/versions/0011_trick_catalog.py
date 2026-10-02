"""Add canonical trick catalog and cross-view video examples.

Revision ID: 0011_trick_catalog
Revises: 0010_judge_technical_clicks
Create Date: 2026-10-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0011_trick_catalog"
down_revision: str | None = "0010_judge_technical_clicks"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "trick_catalog",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "owner_id",
            sa.String(length=36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("division", sa.String(length=4), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("aliases", sa.JSON(), nullable=False),
        sa.Column("description", sa.String(length=2048), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("owner_id", "division", "name", name="uq_trick_catalog_owner_name"),
    )
    op.create_index("ix_trick_catalog_owner_id", "trick_catalog", ["owner_id"])
    op.create_table(
        "trick_examples",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "trick_id",
            sa.String(length=36),
            sa.ForeignKey("trick_catalog.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "video_id",
            sa.String(length=36),
            sa.ForeignKey("video_assets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(length=36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("view_type", sa.String(length=32), nullable=False),
        sa.Column("camera_angle", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("playback_speed", sa.Float(), nullable=False, server_default="1"),
        sa.Column("notes", sa.String(length=2048), nullable=False, server_default=""),
        sa.Column("is_primary", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("start_ms >= 0", name="ck_trick_examples_start_nonnegative"),
        sa.CheckConstraint("end_ms > start_ms", name="ck_trick_examples_end_after_start"),
        sa.CheckConstraint("playback_speed > 0", name="ck_trick_examples_speed_positive"),
    )
    op.create_index("ix_trick_examples_trick_id", "trick_examples", ["trick_id"])
    op.create_index("ix_trick_examples_video_id", "trick_examples", ["video_id"])


def downgrade() -> None:
    op.drop_index("ix_trick_examples_video_id", table_name="trick_examples")
    op.drop_index("ix_trick_examples_trick_id", table_name="trick_examples")
    op.drop_table("trick_examples")
    op.drop_index("ix_trick_catalog_owner_id", table_name="trick_catalog")
    op.drop_table("trick_catalog")
