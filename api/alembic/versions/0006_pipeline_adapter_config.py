"""Add per-analysis pipeline_adapter_config JSON to analysis_jobs.

Revision ID: 0006_pipeline_adapter_config
Revises: 0005_ruleset_version
Create Date: 2026-07-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0006_pipeline_adapter_config"
down_revision: str | None = "0005_ruleset_version"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "analysis_jobs",
        sa.Column("pipeline_adapter_config", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("analysis_jobs", "pipeline_adapter_config")
