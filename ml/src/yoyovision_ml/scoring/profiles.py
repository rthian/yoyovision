"""Prompt D requirement 3: "Support multiple scoring profiles: practice,
judge_assist, research."

A `ScoringProfile` never changes the deterministic math in `scoring_engine`
-- the `Ruleset` remains the single source of truth for point values
(product principle #1). It only changes *which inputs* `scoring.pipeline`
is willing to feed into that math, and which optional stages run.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from yoyovision_ml.domain import ReviewStatus
from yoyovision_ml.ruleset import RepeatedElementPolicyType


class ScoringProfile(StrEnum):
    #: Fast, low-friction self-practice feedback. Model-only/pending
    #: detections count; repeated elements always earn full credit (decay
    #: would discourage the repetition practice requires).
    PRACTICE = "practice"
    #: The default day-to-day profile: a human judge/reviewer is assisted by
    #: model detections and automatic Freestyle-Evaluation estimates, but
    #: nothing requires human confirmation beyond today's existing
    #: reject/confirm workflow.
    JUDGE_ASSIST = "judge_assist"
    #: Strictest profile, for calibration studies and dataset analysis:
    #: only human-confirmed events count, multiple judges are expected, and
    #: the repeated-element decay curve applies to every family (not just
    #: the ruleset's `high_risk_families`).
    RESEARCH = "research"


class ScoringProfileConfig(BaseModel):
    profile: ScoringProfile
    #: Beyond the universal "exclude REJECTED" rule (and the dangerous-play
    #: confirmation gate, which always applies regardless of profile), does
    #: this profile additionally require `review_status == CONFIRMED` for an
    #: event/deduction to count towards scoring?
    require_confirmed_events: bool = False
    #: May `scoring.fe_estimators` auto-fill Freestyle Evaluation categories
    #: the human judge left blank? Never applies to `showmanship` regardless
    #: of this flag -- there is no `estimate_showmanship` function to call.
    use_automatic_fe_estimators: bool = True
    #: Does this profile expect/aggregate multiple human judges'
    #: `JudgeFreestyleScore` entries (`scoring.judges.aggregate_judge_scores`)
    #: rather than a single `FreestyleEvaluation`?
    require_multiple_judges: bool = False
    #: Overrides `ruleset.repeated_element_decay.policy` for this profile
    #: only (the ruleset object itself, and its persisted version string,
    #: are left untouched). `None` defers to whatever the ruleset specifies.
    repeated_element_policy_override: RepeatedElementPolicyType | None = None
    #: Whether `scoring.pipeline.run_scoring_pipeline` computes a bootstrap
    #: uncertainty range for `final_score` (Prompt D requirement 10).
    compute_confidence_interval: bool = True


PROFILE_CONFIGS: dict[ScoringProfile, ScoringProfileConfig] = {
    ScoringProfile.PRACTICE: ScoringProfileConfig(
        profile=ScoringProfile.PRACTICE,
        require_confirmed_events=False,
        use_automatic_fe_estimators=True,
        require_multiple_judges=False,
        repeated_element_policy_override="full_credit",
        compute_confidence_interval=False,
    ),
    ScoringProfile.JUDGE_ASSIST: ScoringProfileConfig(
        profile=ScoringProfile.JUDGE_ASSIST,
        require_confirmed_events=False,
        use_automatic_fe_estimators=True,
        require_multiple_judges=False,
        repeated_element_policy_override=None,
        compute_confidence_interval=True,
    ),
    ScoringProfile.RESEARCH: ScoringProfileConfig(
        profile=ScoringProfile.RESEARCH,
        require_confirmed_events=True,
        use_automatic_fe_estimators=True,
        require_multiple_judges=True,
        repeated_element_policy_override="decay_all_families",
        compute_confidence_interval=True,
    ),
}


def get_profile_config(profile: ScoringProfile) -> ScoringProfileConfig:
    return PROFILE_CONFIGS[profile]


def minimum_review_status_ok(review_status: ReviewStatus, config: ScoringProfileConfig) -> bool:
    """Whether an event/deduction's review status satisfies `config`'s
    `require_confirmed_events` gate, on top of the universal REJECTED
    exclusion applied regardless of profile."""
    if review_status == ReviewStatus.REJECTED:
        return False
    if config.require_confirmed_events:
        return review_status == ReviewStatus.CONFIRMED
    return True
