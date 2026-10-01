"""Add immutable freestyle division provenance.

Revision ID: 0008_multi_division_foundation
Revises: 0007_multi_judge_entries
Create Date: 2026-10-01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0008_multi_division_foundation"
down_revision: str | None = "0007_multi_judge_entries"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    for table_name in ("video_assets", "analysis_jobs", "judging_entries"):
        op.add_column(
            table_name,
            sa.Column("division", sa.String(length=4), nullable=False, server_default="1A"),
        )


def downgrade() -> None:
    for table_name in ("judging_entries", "analysis_jobs", "video_assets"):
        op.drop_column(table_name, "division")
