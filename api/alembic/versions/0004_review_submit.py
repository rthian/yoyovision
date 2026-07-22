"""Add review submit lock columns to analysis_jobs.

Revision ID: 0004_review_submit
Revises: 0003_routine_window
Create Date: 2026-07-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004_review_submit"
down_revision: str | None = "0003_routine_window"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "analysis_jobs",
        sa.Column("review_state", sa.String(length=16), nullable=False, server_default="draft"),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("submitted_by", sa.String(length=36), sa.ForeignKey("users.id"), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("analysis_jobs", "submitted_by")
    op.drop_column("analysis_jobs", "submitted_at")
    op.drop_column("analysis_jobs", "review_state")
