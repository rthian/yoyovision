"""Prompt D requirement 2: "Separate: event detection / technical counting /
technical scaling / Freestyle Evaluation / major deductions" and requirement
10: "Output confidence intervals or uncertainty ranges where supported."

`run_scoring_pipeline` is the one orchestrator that ties together every
other module in this package (`profiles`, `overrides`, `judges`,
`fe_estimators`, `scoring_engine`) into a single, fully audited
`ScoringPipelineResult`. It deliberately does NOT run event detection or
dangerous-play detection itself -- those are upstream stages (Prompt B/C's
event detector, `scoring.dangerous_play.detect_dangerous_play`) that
produce ordinary `AnalysisEvent`/`MajorDeduction` rows *before* this
pipeline ever sees them. A dangerous-play flag is therefore just another
`MajorDeduction` with `review_status=PENDING` on the way in; this module
relies entirely on `scoring_engine.deduction_is_scorable` to keep it from
ever affecting `final_score` until a human confirms it -- it never adds
special-case logic for that one deduction type.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from yoyovision_ml.domain import (
    AnalysisEvent,
    AnalysisEventPrediction,
    DeductionPrediction,
    FeatureSet,
    FreestyleEvaluation,
    MajorDeduction,
    ReviewStatus,
    Source,
)
from yoyovision_ml.ruleset import Ruleset
from yoyovision_ml.scoring import fe_estimators
from yoyovision_ml.scoring.judges import aggregate_judge_scores
from yoyovision_ml.scoring.overrides import apply_overrides
from yoyovision_ml.scoring.profiles import (
    ScoringProfile,
    get_profile_config,
    minimum_review_status_ok,
)
from yoyovision_ml.scoring.types import (
    EventOverride,
    FreestyleEvaluationEstimate,
    JudgeFreestyleScore,
    ScoringPipelineResult,
)
from yoyovision_ml.scoring_engine import (
    DeterministicScoringEngine,
    deduction_is_scorable,
    deduction_points,
    freestyle_evaluation_points,
    technical_points,
)

#: The 7 Freestyle Evaluation categories `fe_estimators` can fill gaps for --
#: mirrors `fe_estimators.estimate_all`'s output keys exactly.
#: `showmanship` is deliberately excluded (Prompt D: "Keep showmanship
#: manual by default" -- there is no `estimate_showmanship`).
_ESTIMATABLE_FE_CATEGORIES: tuple[str, ...] = (
    "execution",
    "control",
    "trick_diversity",
    "space_use_emphasis",
    "music_choreography",
    "music_construction",
    "body_control",
)
#: All 8 categories on `domain.FreestyleEvaluation`, in field order.
_ALL_FE_CATEGORIES: tuple[str, ...] = (*_ESTIMATABLE_FE_CATEGORIES, "showmanship")

_DEFAULT_BOOTSTRAP_ITERATIONS = 500
_CI_LOWER_PERCENTILE = 2.5
_CI_UPPER_PERCENTILE = 97.5


def _event_to_prediction(event: AnalysisEvent) -> AnalysisEventPrediction:
    """Mirrors `api.services.scoring_service._event_to_prediction`, kept as
    an independent definition since `ml` cannot depend on `api`."""
    return AnalysisEventPrediction(
        label=event.label,
        family=event.family,
        start_ms=event.start_ms,
        end_ms=event.end_ms,
        confidence=event.confidence,
        outcome=event.outcome,
        difficulty_band=event.difficulty_band,
        model_name=event.model_name or "human",
        model_version=event.model_version or "n/a",
    )


def _deduction_to_prediction(deduction: MajorDeduction) -> DeductionPrediction:
    """Mirrors `api.services.scoring_service._deduction_to_prediction`."""
    return DeductionPrediction(
        type=deduction.type,
        timestamp_ms=deduction.timestamp_ms,
        quantity=deduction.quantity,
        confidence=deduction.confidence,
        model_name="human" if deduction.source == Source.HUMAN else "model",
        model_version="n/a",
    )


def _effective_ruleset(ruleset: Ruleset, policy_override: str | None) -> Ruleset:
    """Prompt D requirement 5 combined with requirement 3: a profile may
    override which repeated-element policy applies, without mutating the
    shared, versioned `Ruleset` object itself (`profiles.ScoringProfileConfig`:
    "the ruleset object itself... are left untouched")."""
    if policy_override is None:
        return ruleset
    effective = ruleset.model_copy(deep=True)
    effective.repeated_element_decay.policy = policy_override  # type: ignore[assignment]
    return effective


def _fill_fe_gaps(
    evaluation: FreestyleEvaluation | None,
    estimates: dict[str, FreestyleEvaluationEstimate],
) -> tuple[FreestyleEvaluation, bool, list[str]]:
    """Fills any `None` category in `evaluation` from `estimates` (Prompt D
    FREESTYLE EVALUATION section). Never overwrites a human-entered value,
    and never fills `showmanship` (no estimator exists for it). Returns the
    filled evaluation, whether any estimate was actually used, and
    human-readable warnings for each filled category."""
    values: dict[str, float | None] = {}
    warnings: list[str] = []
    used_estimate = False

    for category in _ESTIMATABLE_FE_CATEGORIES:
        current = getattr(evaluation, category) if evaluation is not None else None
        if current is not None:
            values[category] = current
            continue
        estimate = estimates.get(category)
        if estimate is not None and estimate.value is not None:
            values[category] = estimate.value
            used_estimate = True
            warnings.append(
                f"'{category}' has no human value; auto-estimated at "
                f"{estimate.value:.1f}/10 (confidence {estimate.confidence:.0%}). "
                f"{estimate.warning}"
            )
        else:
            values[category] = None

    values["showmanship"] = evaluation.showmanship if evaluation is not None else None
    filled = FreestyleEvaluation(
        **values,
        source=evaluation.source if evaluation is not None else Source.MODEL,
        notes=evaluation.notes if evaluation is not None else "",
    )
    return filled, used_estimate, warnings


def _fe_source_label(has_human_value: bool, used_estimate: bool) -> str:
    if has_human_value and used_estimate:
        return "human+estimated"
    if has_human_value:
        return "human"
    if used_estimate:
        return "estimated"
    return "none"


def _bootstrap_final_score_interval(
    *,
    scorable_events: list[AnalysisEventPrediction],
    scorable_deductions: list[DeductionPrediction],
    judge_scores: Sequence[JudgeFreestyleScore],
    fixed_evaluation: FreestyleEvaluation | None,
    fe_estimates: dict[str, FreestyleEvaluationEstimate],
    ruleset: Ruleset,
    iterations: int,
    seed: int,
) -> tuple[float, float] | None:
    """Prompt D requirement 10: a seeded nonparametric bootstrap over the
    *populations* already fed into the deterministic stage functions --
    never a re-derivation of the scoring math itself, so there is no risk
    of the bootstrap silently drifting from `scoring_engine`'s canonical
    calculation. Each iteration resamples-with-replacement from:

    - the scorable technical events (`technical_points`)
    - the scorable major deductions (`deduction_points`)
    - the judges' Freestyle Evaluation entries, when 2+ judges were
      submitted (`aggregate_judge_scores` + `freestyle_evaluation_points`)

    Returns `None` when there is nothing to resample at all (no events, no
    deductions, and fewer than 2 judges) -- an uncertainty range would be
    meaningless, not just wide.
    """
    if not scorable_events and not scorable_deductions and len(judge_scores) < 2:
        return None
    if iterations <= 0:
        return None

    rng = np.random.default_rng(seed)
    final_scores = np.empty(iterations, dtype=float)

    for i in range(iterations):
        if scorable_events:
            resample_idx = rng.integers(0, len(scorable_events), size=len(scorable_events))
            resampled_events = [scorable_events[j] for j in resample_idx]
            technical_raw_i, _ = technical_points(resampled_events, ruleset)
        else:
            technical_raw_i = 0.0
        technical_scaled_i = min(technical_raw_i, ruleset.technical_scale_max)

        if scorable_deductions:
            resample_idx = rng.integers(0, len(scorable_deductions), size=len(scorable_deductions))
            resampled_deductions = [scorable_deductions[j] for j in resample_idx]
            deduction_total_i, _ = deduction_points(resampled_deductions, ruleset)
        else:
            deduction_total_i = 0.0

        if len(judge_scores) >= 2:
            resample_idx = rng.integers(0, len(judge_scores), size=len(judge_scores))
            resampled_judges = [judge_scores[j] for j in resample_idx]
            aggregated_i, _ = aggregate_judge_scores(resampled_judges)
            filled_i, _, _ = _fill_fe_gaps(aggregated_i, fe_estimates)
            _, fe_scaled_i, _ = freestyle_evaluation_points(filled_i, ruleset)
        else:
            _, fe_scaled_i, _ = freestyle_evaluation_points(fixed_evaluation, ruleset)

        final_scores[i] = max(
            0.0,
            ruleset.technical_weight * technical_scaled_i
            + ruleset.freestyle_evaluation_weight * fe_scaled_i
            - deduction_total_i,
        )

    lower, upper = np.percentile(final_scores, [_CI_LOWER_PERCENTILE, _CI_UPPER_PERCENTILE])
    return round(float(lower), 3), round(float(upper), 3)


def run_scoring_pipeline(
    *,
    events: Sequence[AnalysisEvent],
    deductions: Sequence[MajorDeduction],
    ruleset: Ruleset,
    profile: ScoringProfile = ScoringProfile.JUDGE_ASSIST,
    human_evaluation: FreestyleEvaluation | None = None,
    judge_scores: Sequence[JudgeFreestyleScore] = (),
    event_overrides: Sequence[EventOverride] = (),
    features: FeatureSet | None = None,
    bootstrap_iterations: int = _DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = 0,
) -> ScoringPipelineResult:
    """Runs the full Prompt D scoring pipeline for one routine and returns a
    fully auditable `ScoringPipelineResult`.

    Stage order (requirement 2's separation, enforced by construction --
    each stage below calls a distinct, independently testable function):
    1. apply per-event manual overrides (`overrides.apply_overrides`)
    2. filter events/deductions by review status per `profile`
       (`profiles.minimum_review_status_ok`) and by the manual-confirmation
       gate (`scoring_engine.deduction_is_scorable`) -- always applied,
       regardless of profile, per requirement on dangerous-play review
    3. aggregate multiple judges' Freestyle Evaluation entries, if given
       (`judges.aggregate_judge_scores`), else use `human_evaluation`
    4. fill any remaining Freestyle Evaluation gaps with automatic
       estimators, if the profile allows it (`fe_estimators.estimate_all`)
    5. technical counting + scaling, major deductions, and Freestyle
       Evaluation scaling (`scoring_engine`'s stage functions, via
       `DeterministicScoringEngine.calculate` for the point estimate)
    6. optionally, a seeded bootstrap confidence interval around
       `final_score` (requirement 10)

    Exactly one of `human_evaluation`/`judge_scores` is normally supplied;
    if both are, `judge_scores` (once aggregated) takes precedence as the
    more complete signal, and `human_evaluation` is ignored with a warning.
    """
    warnings: list[str] = []
    config = get_profile_config(profile)
    effective_ruleset = _effective_ruleset(ruleset, config.repeated_element_policy_override)

    if judge_scores and human_evaluation is not None:
        warnings.append(
            "Both judge_scores and human_evaluation were provided; using the "
            "aggregated judge_scores and ignoring human_evaluation."
        )
    if config.require_multiple_judges and len(judge_scores) < 2:
        warnings.append(
            f"Profile '{profile.value}' expects multiple judge scores; only "
            f"{len(judge_scores)} were provided."
        )

    corrected_events, override_audit_log = apply_overrides(events, event_overrides)

    events_for_scoring = [
        e for e in corrected_events if minimum_review_status_ok(e.review_status, config)
    ]
    event_predictions = [_event_to_prediction(e) for e in events_for_scoring]

    deductions_for_scoring = [
        d
        for d in deductions
        if minimum_review_status_ok(d.review_status, config)
        and deduction_is_scorable(d.type, d.review_status, effective_ruleset)
    ]
    deduction_predictions = [_deduction_to_prediction(d) for d in deductions_for_scoring]

    deductions_awaiting_confirmation = 0
    for deduction in deductions:
        if deduction.review_status in (ReviewStatus.REJECTED, ReviewStatus.CONFIRMED):
            continue
        rule = effective_ruleset.deduction_rule_for(deduction.type)
        if rule is not None and rule.requires_manual_confirmation:
            deductions_awaiting_confirmation += 1

    base_evaluation: FreestyleEvaluation | None
    if judge_scores:
        base_evaluation, judge_warnings = aggregate_judge_scores(judge_scores)
        warnings.extend(judge_warnings)
    else:
        base_evaluation = human_evaluation

    has_human_value = base_evaluation is not None and any(
        getattr(base_evaluation, category) is not None for category in _ALL_FE_CATEGORIES
    )

    fe_estimates: dict[str, FreestyleEvaluationEstimate] = {}
    used_estimate = False
    evaluation_for_scoring: FreestyleEvaluation | None
    if config.use_automatic_fe_estimators:
        fe_estimates = fe_estimators.estimate_all(event_predictions, features)
        evaluation_for_scoring, used_estimate, estimate_warnings = _fill_fe_gaps(
            base_evaluation, fe_estimates
        )
        warnings.extend(estimate_warnings)
    else:
        evaluation_for_scoring = base_evaluation

    fe_source = _fe_source_label(has_human_value, used_estimate)

    breakdown = DeterministicScoringEngine().calculate(
        events=event_predictions,
        deductions=deduction_predictions,
        freestyle_evaluation=evaluation_for_scoring,
        ruleset=effective_ruleset,
    )

    final_score_interval: tuple[float, float] | None = None
    if config.compute_confidence_interval:
        final_score_interval = _bootstrap_final_score_interval(
            scorable_events=event_predictions,
            scorable_deductions=deduction_predictions,
            judge_scores=judge_scores,
            fixed_evaluation=evaluation_for_scoring,
            fe_estimates=fe_estimates,
            ruleset=effective_ruleset,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        if final_score_interval is None:
            warnings.append(
                "Not enough independent samples (events, deductions, or judges) "
                "to compute a bootstrap confidence interval for final_score."
            )

    return ScoringPipelineResult(
        profile=profile.value,
        ruleset_version=ruleset.version,
        technical_event_count=len(event_predictions),
        technical_raw=breakdown.technical_raw,
        technical_scaled=breakdown.technical_scaled,
        deduction_count=sum(d.quantity for d in deduction_predictions),
        deductions_awaiting_confirmation=deductions_awaiting_confirmation,
        major_deductions=breakdown.major_deductions,
        freestyle_evaluation_raw=breakdown.freestyle_evaluation_raw,
        freestyle_evaluation_scaled=breakdown.freestyle_evaluation_scaled,
        freestyle_evaluation_source=fe_source,
        fe_estimates=tuple(fe_estimates.values()),
        override_audit_log=tuple(override_audit_log),
        breakdown=breakdown,
        final_score_interval=final_score_interval,
        warnings=[*warnings, *breakdown.warnings],
    )
