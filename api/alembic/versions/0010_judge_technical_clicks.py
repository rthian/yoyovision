"""Add timestamped human technical judging clicks.

Revision ID: 0010_judge_technical_clicks
Revises: 0009_training_data_workflow
Create Date: 2026-10-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0010_judge_technical_clicks"
down_revision: str | None = "0009_training_data_workflow"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "judge_technical_clicks",
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
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_judge_technical_clicks_assignment_id",
        "judge_technical_clicks",
        ["assignment_id"],
    )
    op.create_index(
        "ix_judge_technical_clicks_entry_video_id",
        "judge_technical_clicks",
        ["entry_video_id"],
    )


def downgrade() -> None:
    op.drop_table("judge_technical_clicks")
