"""Prompt F (production inference): adds observability/control columns to
`analysis_jobs` -- model versions, device, runtime, per-stage timings,
shadow-mode flag, cooperative cancellation flag, and retry count.

Revision ID: 0002_prompt_f_inference
Revises: 0001_initial
Create Date: 2026-07-21

Note: `alembic_version.version_num` is `VARCHAR(32)` (Alembic's own
default), so this revision id is deliberately kept under that limit --
`"0002_prompt_f_production_inference"` (35 chars) does not fit.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_prompt_f_inference"
down_revision: str | None = "0001_initial"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("analysis_jobs", sa.Column("model_versions", sa.JSON(), nullable=True))
    op.add_column("analysis_jobs", sa.Column("device", sa.String(length=32), nullable=True))
    op.add_column("analysis_jobs", sa.Column("runtime_versions", sa.JSON(), nullable=True))
    op.add_column("analysis_jobs", sa.Column("stage_durations_ms", sa.JSON(), nullable=True))
    op.add_column(
        "analysis_jobs",
        sa.Column("is_shadow", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "analysis_jobs",
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("analysis_jobs", "retry_count")
    op.drop_column("analysis_jobs", "cancel_requested")
    op.drop_column("analysis_jobs", "is_shadow")
    op.drop_column("analysis_jobs", "stage_durations_ms")
    op.drop_column("analysis_jobs", "runtime_versions")
    op.drop_column("analysis_jobs", "device")
    op.drop_column("analysis_jobs", "model_versions")
