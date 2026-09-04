"""Add click_mode and judge_clicks table (Phase F clicker v2).

Revision ID: 0008_judge_clicks
Revises: 0007_multi_judge_entries
Create Date: 2026-08-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0008_judge_clicks"
down_revision: str | None = "0007_multi_judge_entries"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "judging_entries",
        sa.Column("click_mode", sa.String(length=24), nullable=False, server_default="off"),
    )

    op.create_table(
        "judge_clicks",
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
        sa.Column("timestamp_ms", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_judge_clicks_assignment_id", "judge_clicks", ["assignment_id"])
    op.create_index(
        "ix_judge_clicks_entry_video_id", "judge_clicks", ["entry_video_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_judge_clicks_entry_video_id", table_name="judge_clicks")
    op.drop_index("ix_judge_clicks_assignment_id", table_name="judge_clicks")
    op.drop_table("judge_clicks")
    op.drop_column("judging_entries", "click_mode")
