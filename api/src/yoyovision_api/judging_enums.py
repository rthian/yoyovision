"""Enums for multi-judge video entries (Phase B)."""

from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    USER = "user"
    ADMIN = "admin"


class JudgingEntryMode(StrEnum):
    TRAINING = "training"
    CONTEST = "contest"


class JudgingEntryStatus(StrEnum):
    DRAFT = "draft"
    OPEN = "open"
    LOCKED = "locked"


class AiMixProfile(StrEnum):
    COMPARE_ONLY = "A"
    GAP_FILL = "B"
    EQUAL_VOTE = "C"


class AggregationMode(StrEnum):
    AUTO = "auto"
    SIMPLE_MEAN = "simple_mean"
    TRIM_1 = "trim_1"
    TRIM_2 = "trim_2"


class JudgeAssignmentStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"

class ClickMode(StrEnum):
    """Phase F: timestamp clicker behavior for judging entries."""

    OFF = "off"
    TRAINING_ONLY = "training_only"
    TECHNICAL_SCORE = "technical_score"
