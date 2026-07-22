"""Add optional routine window bounds to analysis_jobs.

Revision ID: 0003_routine_window
Revises: 0002_prompt_f_inference
Create Date: 2026-07-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_routine_window"
down_revision: str | None = "0002_prompt_f_inference"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("analysis_jobs", sa.Column("routine_start_ms", sa.Integer(), nullable=True))
    op.add_column("analysis_jobs", sa.Column("routine_end_ms", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("analysis_jobs", "routine_end_ms")
    op.drop_column("analysis_jobs", "routine_start_ms")
