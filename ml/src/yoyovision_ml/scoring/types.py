"""Prompt D domain types: judge clicks/scores, per-event manual overrides,
automatic Freestyle-Evaluation estimates, and the multi-stage pipeline
result.

Kept out of the shared `yoyovision_ml.domain` module (unlike
`AnalysisEvent`/`MajorDeduction`) because none of these are persisted by
`api`'s ORM today -- see this package's `__init__.py` docstring for the
current, deliberately `ml`-only scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from yoyovision_ml.domain import ScoreBreakdown


@dataclass(slots=True, frozen=True)
class JudgeClick:
    """A raw, low-effort timestamp click from a human judge watching the
    routine live or on review (Prompt D requirement 7: "Support manually
    entered judge clicks"). Mirrors `dataset.schema.JudgeClickAnnotation`
    intentionally -- see that class's docstring -- but is kept as an
    independent definition here so this live-scoring package never has to
    import the Pydantic dataset-annotation schema just to accept a
    runtime-entered click."""

    click_id: str
    judge_id: str
    timestamp_ms: int
    associated_label: str | None = None
    notes: str = ""


@dataclass(slots=True, frozen=True)
class JudgeFreestyleScore:
    """One judge's Freestyle Evaluation entry (Prompt D requirement 8:
    "Support multiple human judge scores"). `domain.FreestyleEvaluation`
    itself has no `judge_id` field, since it is the *single* evaluation
    `scoring_engine.DeterministicScoringEngine` consumes --
    `scoring.judges.aggregate_judge_scores` reduces a list of these into one
    `FreestyleEvaluation`."""

    judge_id: str
    execution: float | None = None
    control: float | None = None
    trick_diversity: float | None = None
    space_use_emphasis: float | None = None
    music_choreography: float | None = None
    music_construction: float | None = None
    body_control: float | None = None
    showmanship: float | None = None
    notes: str = ""


#: Fields on an `AnalysisEvent`/`AnalysisEventPrediction` a human reviewer is
#: allowed to manually override through `scoring.overrides.apply_overrides`.
OVERRIDABLE_EVENT_FIELDS: frozenset[str] = frozenset(
    {"label", "family", "start_ms", "end_ms", "outcome", "difficulty_band", "confidence"}
)


@dataclass(slots=True, frozen=True)
class EventOverride:
    """A human reviewer's manual correction of one persisted `AnalysisEvent`
    field, keeping the original value for audit (Prompt D requirement 6:
    "Support per-event manual overrides"). `field_name` must be one of
    `OVERRIDABLE_EVENT_FIELDS`. `original_value`/`overridden_value` are
    stored as strings (mirroring how the value would be entered in a UI
    form / logged in an audit record) and are parsed back to the field's
    real type by `scoring.overrides.apply_overrides`."""

    event_id: str
    field_name: str
    original_value: str
    overridden_value: str
    overridden_by: str
    overridden_at: datetime
    reason: str = ""


@dataclass(slots=True, frozen=True)
class FreestyleEvaluationEstimate:
    """One automatically-estimated Freestyle Evaluation category (Prompt D
    "FREESTYLE EVALUATION" section: "All automatically generated FE
    categories must expose: confidence, supporting features, model version,
    warning that artistic scoring is subjective"). `value=None` means the
    estimator deliberately declined to guess (e.g. no audio-analysis stage
    exists for the music categories) -- never a fabricated number."""

    category: str
    value: float | None
    confidence: float
    supporting_features: dict[str, float]
    model_name: str
    model_version: str
    warning: str


@dataclass(slots=True)
class ScoringPipelineResult:
    """Prompt D requirement 2: "Separate: event detection / technical
    counting / technical scaling / Freestyle Evaluation / major deductions."
    Every stage's output is kept on this object so a reviewer (or a test)
    can see exactly why the final score came out the way it did --
    "preserving a complete audit trail" per the prompt's OBJECTIVE."""

    profile: str
    ruleset_version: str
    technical_event_count: int
    technical_raw: float
    technical_scaled: float
    deduction_count: int
    deductions_awaiting_confirmation: int
    major_deductions: float
    freestyle_evaluation_raw: float
    freestyle_evaluation_scaled: float
    #: "human" | "estimated" | "human+estimated" | "none"
    freestyle_evaluation_source: str
    fe_estimates: tuple[FreestyleEvaluationEstimate, ...]
    override_audit_log: tuple[str, ...]
    breakdown: ScoreBreakdown
    #: (lower, upper) bootstrap uncertainty range for `breakdown.final_score`,
    #: or None when the active profile disabled it (requirement 10).
    final_score_interval: tuple[float, float] | None = None
    warnings: list[str] = field(default_factory=list)
