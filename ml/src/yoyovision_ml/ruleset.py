"""Versioned, configurable scoring ruleset.

IMPORTANT: YoYoVision ships with a first-draft, unofficial ruleset
(`1a-draft-0.1`) modeled loosely on the publicly known structure of 1A
freestyle judging (technical elements, major deductions, freestyle
categories). It is NOT sourced from, endorsed by, or certified by IYYF,
WYYC, or any competition body. See docs/ruleset.md. Organizations may
supply their own ruleset YAML file and version it independently.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from yoyovision_ml.domain import DeductionType, DifficultyBand, EventFamily

_RULESETS_DIR = Path(__file__).parent / "rulesets"
_DEFAULT_RULESET_FILENAME = "1a_draft_0_1.yaml"

#: Prompt D: "Support repeated-element policies" (plural). Each policy
#: answers "does the Nth occurrence of this element get full credit?":
#:   - "decay_high_risk_only": today's/`judge_assist`'s default -- only
#:     `high_risk_families` decay via `occurrence_multipliers`; everything
#:     else earns full credit every time.
#:   - "decay_all_families": the same curve applies to *every* family, not
#:     just `high_risk_families` -- a stricter, research-oriented policy.
#:   - "full_credit": decay disabled outright (e.g. `practice` mode, where
#:     discouraging repetition is counterproductive).
#:   - "cap_occurrences": like `decay_high_risk_only`, but occurrences beyond
#:     `len(occurrence_multipliers)` earn *zero* credit instead of clamping
#:     to the last multiplier -- a hard repeat cap.
RepeatedElementPolicyType = Literal[
    "decay_high_risk_only", "decay_all_families", "full_credit", "cap_occurrences"
]


class RepeatedElementDecay(BaseModel):
    """Reduced credit for repeated high-risk / repeated elements (product req)."""

    #: multiplier applied to the Nth occurrence of the *same* (family, label)
    #: combination within a routine, 1-indexed. Index beyond the list length
    #: reuses the last value (except under the `"cap_occurrences"` policy,
    #: where it becomes 0.0 instead).
    occurrence_multipliers: list[float] = Field(default=[1.0, 0.7, 0.4, 0.2])
    #: families considered "high risk" and therefore eligible for decay under
    #: the `"decay_high_risk_only"`/`"cap_occurrences"` policies; repeats
    #: outside this set are not penalized under those two policies.
    high_risk_families: list[EventFamily] = Field(
        default=[
            EventFamily.SUICIDE,
            EventFamily.WHIP_CATCH,
            EventFamily.HORIZONTAL,
        ]
    )
    #: Which of `RepeatedElementPolicyType` this ruleset (or scoring profile
    #: override -- see `scoring.profiles`) applies. Defaults to today's
    #: pre-Prompt-D behavior so existing rulesets/tests are unaffected.
    policy: RepeatedElementPolicyType = "decay_high_risk_only"

    def multiplier_for_occurrence(self, occurrence_index: int) -> float:
        """`occurrence_index` is 1-indexed (1 = first time this element is seen).

        Kept as its own method (rather than folded into `multiplier_for`) since
        existing callers/tests reference it directly for the un-capped decay
        curve regardless of which family is involved.
        """
        if occurrence_index <= 0:
            raise ValueError("occurrence_index must be >= 1")
        idx = min(occurrence_index, len(self.occurrence_multipliers)) - 1
        return self.occurrence_multipliers[idx]

    def multiplier_for(
        self,
        family: EventFamily,
        occurrence_index: int,
        policy: RepeatedElementPolicyType | None = None,
    ) -> float:
        """The credit multiplier for the `occurrence_index`-th time `family`
        (paired with a specific label, tracked by the caller) has been seen,
        under `policy` (defaults to `self.policy`)."""
        if occurrence_index <= 0:
            raise ValueError("occurrence_index must be >= 1")
        effective_policy = policy or self.policy

        if effective_policy == "full_credit":
            return 1.0
        if effective_policy == "decay_all_families":
            return self.multiplier_for_occurrence(occurrence_index)
        if effective_policy == "cap_occurrences":
            if family not in self.high_risk_families:
                return 1.0
            if occurrence_index > len(self.occurrence_multipliers):
                return 0.0
            return self.multiplier_for_occurrence(occurrence_index)
        # "decay_high_risk_only" (default)
        if family not in self.high_risk_families:
            return 1.0
        return self.multiplier_for_occurrence(occurrence_index)


class DeductionRule(BaseModel):
    type: DeductionType
    points_per_occurrence: float
    #: if set, deductions of this type beyond this count no longer add further
    #: penalty (protects against runaway penalties from detector noise).
    max_occurrences_penalized: int | None = None
    #: Prompt D: "Dangerous-play detection must never automatically
    #: disqualify a player. It must create a review flag." When True, an
    #: occurrence of this deduction type contributes zero score impact until
    #: a human has explicitly set its `review_status` to `CONFIRMED` --
    #: `ReviewStatus.PENDING` (the default for a freshly-flagged detection)
    #: is *not* enough. See `scoring_engine.deduction_is_scorable`.
    requires_manual_confirmation: bool = False


class DifficultyBandPoints(BaseModel):
    basic: float = 1.0
    intermediate: float = 2.0
    advanced: float = 3.0
    unknown: float = 1.0

    def points_for(self, band: DifficultyBand) -> float:
        return {
            DifficultyBand.BASIC: self.basic,
            DifficultyBand.INTERMEDIATE: self.intermediate,
            DifficultyBand.ADVANCED: self.advanced,
            DifficultyBand.UNKNOWN: self.unknown,
        }[band]


class FreestyleEvaluationWeights(BaseModel):
    execution: float = 1.0
    control: float = 1.0
    trick_diversity: float = 1.0
    space_use_emphasis: float = 1.0
    music_choreography: float = 1.0
    music_construction: float = 1.0
    body_control: float = 1.0
    showmanship: float = 1.0


class Ruleset(BaseModel):
    version: str
    is_official: bool = False
    disclaimer: str = (
        "This ruleset is an unofficial, editable draft used for training and "
        "judge-assistance purposes only. It is not certified by IYYF, WYYC, "
        "or any competition body."
    )
    #: Prompt D requirement 4: "Never label a custom trick difficulty value
    #: as an official IYYF value." Structural enforcement (not just a
    #: warning string): `is_official=True` is rejected unless this cites a
    #: concrete certification source, so it is impossible for a ruleset to
    #: silently claim official status. No packaged ruleset in this
    #: repository sets either field -- see docs/ruleset.md.
    iyyf_certification_reference: str | None = None
    difficulty_band_points: DifficultyBandPoints = Field(default_factory=DifficultyBandPoints)
    repeated_element_decay: RepeatedElementDecay = Field(default_factory=RepeatedElementDecay)
    deduction_rules: list[DeductionRule] = Field(default_factory=list)
    freestyle_evaluation_weights: FreestyleEvaluationWeights = Field(
        default_factory=FreestyleEvaluationWeights
    )
    technical_scale_max: float = 100.0
    freestyle_evaluation_scale_max: float = 100.0
    #: weight of technical vs freestyle-evaluation in the final blended score, must sum to 1.0
    technical_weight: float = 0.6
    freestyle_evaluation_weight: float = 0.4
    #: events below this confidence are always flagged for human review
    low_confidence_review_threshold: float = 0.55

    @field_validator("freestyle_evaluation_weight")
    @classmethod
    def _weights_sum_to_one(cls, v: float, info: Any) -> float:
        technical_weight = info.data.get("technical_weight", 0.6)
        if abs((technical_weight + v) - 1.0) > 1e-6:
            raise ValueError("technical_weight + freestyle_evaluation_weight must equal 1.0")
        return v

    @model_validator(mode="after")
    def _official_requires_certification_reference(self) -> Ruleset:
        if self.is_official and not (self.iyyf_certification_reference or "").strip():
            raise ValueError(
                "is_official=True requires a non-empty iyyf_certification_reference "
                "citing where the official values came from -- a ruleset must never "
                "silently claim IYYF-official status (Prompt D requirement 4)."
            )
        return self

    def deduction_rule_for(self, deduction_type: DeductionType) -> DeductionRule | None:
        for rule in self.deduction_rules:
            if rule.type == deduction_type:
                return rule
        return None


def load_ruleset(path: Path) -> Ruleset:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Ruleset.model_validate(raw)


@lru_cache(maxsize=8)
def default_ruleset() -> Ruleset:
    """The packaged first-draft 1A ruleset. See module docstring disclaimer."""
    return load_ruleset(_RULESETS_DIR / _DEFAULT_RULESET_FILENAME)


def available_ruleset_files() -> list[Path]:
    if not _RULESETS_DIR.exists():
        return []
    return sorted(_RULESETS_DIR.glob("*.yaml"))


def list_available_rulesets() -> list[Ruleset]:
    """Loads every packaged ruleset file, for transparency endpoints that let
    users inspect exactly which versioned config produced a given score."""
    return [load_ruleset(path) for path in available_ruleset_files()]


def get_ruleset_by_version(version: str) -> Ruleset | None:
    for ruleset in list_available_rulesets():
        if ruleset.version == version:
            return ruleset
    return None
