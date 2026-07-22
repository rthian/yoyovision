"""Add per-analysis ruleset_version to analysis_jobs.

Revision ID: 0005_ruleset_version
Revises: 0004_review_submit
Create Date: 2026-07-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005_ruleset_version"
down_revision: str | None = "0004_review_submit"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "analysis_jobs",
        sa.Column(
            "ruleset_version",
            sa.String(length=32),
            nullable=False,
            server_default="1a-draft-0.1",
        ),
    )


def downgrade() -> None:
    op.drop_column("analysis_jobs", "ruleset_version")
