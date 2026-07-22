"""Deterministic, rules-based scoring engine.

Per product principle #1, this NEVER predicts a final score directly from an
opaque model. It only consumes already-detected events/deductions (each with
timestamps, confidence, and evidence) and applies a versioned, configurable
`Ruleset` to compute a fully auditable `ScoreBreakdown`.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from yoyovision_ml.domain import (
    MISTAKE_EVENT_FAMILIES,
    POSITIVE_EVENT_FAMILIES,
    AnalysisEventPrediction,
    DeductionPrediction,
    DeductionType,
    EventFamily,
    FreestyleEvaluation,
    Outcome,
    ReviewStatus,
    ScoreBreakdown,
    TechnicalLineItem,
)
from yoyovision_ml.ruleset import Ruleset

_UNOFFICIAL_WARNING = (
    "This score is an unofficial estimate for training/judge-assistance only. "
    "It is not certified by IYYF, WYYC, or any competition body."
)


def _element_key(event: AnalysisEventPrediction) -> tuple[EventFamily, str]:
    return (event.family, event.label)


def _technical_points_with_line_items(
    events: list[AnalysisEventPrediction],
    ruleset: Ruleset,
    *,
    event_ids: list[str] | None = None,
) -> tuple[float, list[str], list[TechnicalLineItem]]:
    """Sum positive credit for successful technical elements, applying the
    repeated-high-risk-element decay curve. Also returns one audit row per
    input event so the review UI can show per-trick points."""
    warnings: list[str] = []
    occurrence_counts: Counter[tuple[EventFamily, str]] = Counter()
    total = 0.0
    line_items: list[TechnicalLineItem] = []

    sorted_pairs = sorted(
        (
            (event_ids[index] if event_ids is not None and index < len(event_ids) else None, event)
            for index, event in enumerate(events)
        ),
        key=lambda pair: pair[1].start_ms,
    )
    for event_id, event in sorted_pairs:
        if event.family in MISTAKE_EVENT_FAMILIES:
            line_items.append(
                TechnicalLineItem(
                    event_id=event_id,
                    start_ms=event.start_ms,
                    label=event.label,
                    family=event.family,
                    base_points=0.0,
                    multiplier=0.0,
                    points=0.0,
                    reason="excluded_mistake",
                )
            )
            continue
        if event.family not in POSITIVE_EVENT_FAMILIES:
            reason = (
                "excluded_unknown"
                if event.family == EventFamily.UNKNOWN_TECHNICAL_ELEMENT
                else "excluded_equipment"
            )
            if event.family == EventFamily.UNKNOWN_TECHNICAL_ELEMENT:
                warnings.append(
                    f"Unclassified technical element at {event.start_ms}ms requires review."
                )
            line_items.append(
                TechnicalLineItem(
                    event_id=event_id,
                    start_ms=event.start_ms,
                    label=event.label,
                    family=event.family,
                    base_points=0.0,
                    multiplier=0.0,
                    points=0.0,
                    reason=reason,
                )
            )
            continue
        if event.outcome != Outcome.SUCCESS:
            reason = (
                "excluded_uncertain"
                if event.outcome == Outcome.UNCERTAIN
                else "excluded_outcome_miss"
            )
            if event.outcome == Outcome.UNCERTAIN:
                warnings.append(
                    f"Uncertain outcome for '{event.label}' at {event.start_ms}ms "
                    "excluded from technical credit pending review."
                )
            line_items.append(
                TechnicalLineItem(
                    event_id=event_id,
                    start_ms=event.start_ms,
                    label=event.label,
                    family=event.family,
                    base_points=0.0,
                    multiplier=0.0,
                    points=0.0,
                    reason=reason,
                )
            )
            continue

        key = _element_key(event)
        occurrence_counts[key] += 1
        occurrence_index = occurrence_counts[key]

        base_points = ruleset.difficulty_band_points.points_for(event.difficulty_band)
        decay = ruleset.repeated_element_decay
        multiplier = decay.multiplier_for(event.family, occurrence_index)
        credited = base_points * multiplier
        reason = "credited"
        if multiplier < 1.0:
            reason = f"repeat_occurrence_{occurrence_index}"
            if decay.policy in ("decay_high_risk_only", "cap_occurrences") and (
                event.family in decay.high_risk_families
            ):
                warnings.append(
                    f"Repeated high-risk element '{event.label}' "
                    f"(occurrence {occurrence_index}) credited at {multiplier:.0%}."
                )
            else:
                warnings.append(
                    f"Repeated element '{event.label}' (occurrence {occurrence_index}) "
                    f"credited at {multiplier:.0%} under the '{decay.policy}' "
                    "repeated-element policy."
                )
        total += credited
        line_items.append(
            TechnicalLineItem(
                event_id=event_id,
                start_ms=event.start_ms,
                label=event.label,
                family=event.family,
                base_points=base_points,
                multiplier=multiplier,
                points=credited,
                reason=reason,
            )
        )

    return total, warnings, line_items


def _technical_points(
    events: list[AnalysisEventPrediction], ruleset: Ruleset
) -> tuple[float, list[str]]:
    """Sum positive credit for successful technical elements, applying the
    repeated-high-risk-element decay curve. Misses and non-success outcomes
    never earn positive credit (per product requirement: positive vs
    negative technical events are handled distinctly)."""
    total, warnings, _ = _technical_points_with_line_items(events, ruleset)
    return total, warnings


def technical_line_items(
    events: list[AnalysisEventPrediction],
    ruleset: Ruleset,
    *,
    event_ids: list[str] | None = None,
) -> tuple[float, list[str], list[TechnicalLineItem]]:
    """Public entry point for per-event technical credit audit rows."""
    return _technical_points_with_line_items(events, ruleset, event_ids=event_ids)


def technical_points(
    events: list[AnalysisEventPrediction], ruleset: Ruleset
) -> tuple[float, list[str]]:
    """Public entry point for the "technical counting" stage in isolation
    (Prompt D requirement 2: "Separate: event detection / technical
    counting / technical scaling / ..."), e.g. for
    `scoring.pipeline`'s bootstrap confidence-interval resampling, without
    running the rest of `DeterministicScoringEngine.calculate`."""
    return _technical_points(events, ruleset)


def deduction_points(
    deductions: list[DeductionPrediction], ruleset: Ruleset
) -> tuple[float, list[str]]:
    """Public entry point for the "major deductions" stage in isolation."""
    return _deduction_points(deductions, ruleset)


def freestyle_evaluation_points(
    evaluation: FreestyleEvaluation | None, ruleset: Ruleset
) -> tuple[float, float, list[str]]:
    """Public entry point for the "Freestyle Evaluation" stage in isolation."""
    return _freestyle_evaluation_points(evaluation, ruleset)


def deduction_is_scorable(
    deduction_type: DeductionType, review_status: ReviewStatus, ruleset: Ruleset
) -> bool:
    """Whether a *persisted* deduction (which has a `review_status`, unlike
    the pure `DeductionPrediction` this module otherwise operates on) should
    be included when a caller builds the `list[DeductionPrediction]` passed
    to `DeterministicScoringEngine.calculate`.

    Rejected deductions are never scorable (existing behavior). Deduction
    types whose ruleset rule sets `requires_manual_confirmation=True` (e.g.
    `DANGEROUS_PLAY_REVIEW`) are additionally excluded until a human has
    explicitly set `review_status=CONFIRMED` -- per Prompt D's "Dangerous-play
    detection must never automatically disqualify a player. It must create a
    review flag": a freshly-detected dangerous-play flag (`PENDING`)
    contributes zero score impact by construction, not by convention.

    Callers (e.g. `api.services.scoring_service.recompute_score`) are
    expected to filter with this function *before* converting ORM rows into
    `DeductionPrediction`s -- see that module's docstring.
    """
    if review_status == ReviewStatus.REJECTED:
        return False
    rule = ruleset.deduction_rule_for(deduction_type)
    if rule is None or not rule.requires_manual_confirmation:
        return True
    return review_status == ReviewStatus.CONFIRMED


def _deduction_points(
    deductions: list[DeductionPrediction], ruleset: Ruleset
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    quantity_used_by_type: dict[DeductionType, int] = defaultdict(int)
    total = 0.0

    for deduction in sorted(deductions, key=lambda d: d.timestamp_ms):
        rule = ruleset.deduction_rule_for(deduction.type)
        if rule is None:
            warnings.append(f"No deduction rule configured for '{deduction.type}'; skipped.")
            continue

        if deduction.points is not None:
            row_points = deduction.points
        else:
            row_points = rule.points_per_occurrence * deduction.quantity

        allowed_quantity = deduction.quantity
        if rule.max_occurrences_penalized is not None:
            cap = rule.max_occurrences_penalized
            prior = quantity_used_by_type[deduction.type]
            allowed_quantity = max(0, min(deduction.quantity, cap - prior))
            quantity_used_by_type[deduction.type] += deduction.quantity
            if allowed_quantity == 0:
                warnings.append(
                    f"'{deduction.type}' at {deduction.timestamp_ms}ms exceeds the ruleset "
                    f"cap of {cap}; skipped."
                )
                continue
            if allowed_quantity < deduction.quantity:
                row_points = row_points * (allowed_quantity / deduction.quantity)
                warnings.append(
                    f"'{deduction.type}' at {deduction.timestamp_ms}ms only "
                    f"{allowed_quantity} of {deduction.quantity} occurrence(s) penalized "
                    "per ruleset cap."
                )
        else:
            quantity_used_by_type[deduction.type] += deduction.quantity

        total += row_points

    return total, warnings


def _freestyle_evaluation_points(
    evaluation: FreestyleEvaluation | None, ruleset: Ruleset
) -> tuple[float, float, list[str]]:
    weights = ruleset.freestyle_evaluation_weights
    max_possible = sum(
        [
            weights.execution,
            weights.control,
            weights.trick_diversity,
            weights.space_use_emphasis,
            weights.music_choreography,
            weights.music_construction,
            weights.body_control,
            weights.showmanship,
        ]
    )

    if evaluation is None:
        return (
            0.0,
            0.0,
            [
                "Freestyle Evaluation not provided; using placeholder value of 0. "
                "A human judge must enter manual Freestyle Evaluation scores."
            ],
        )

    fields = [
        (evaluation.execution, weights.execution),
        (evaluation.control, weights.control),
        (evaluation.trick_diversity, weights.trick_diversity),
        (evaluation.space_use_emphasis, weights.space_use_emphasis),
        (evaluation.music_choreography, weights.music_choreography),
        (evaluation.music_construction, weights.music_construction),
        (evaluation.body_control, weights.body_control),
        (evaluation.showmanship, weights.showmanship),
    ]

    warnings: list[str] = []
    raw = 0.0
    missing = 0
    for value, weight in fields:
        if value is None:
            missing += 1
            continue
        raw += value * weight

    if missing:
        warnings.append(
            f"{missing} of 8 Freestyle Evaluation categories are missing manual values."
        )

    scaled = 0.0
    if max_possible > 0:
        # each category assumed entered on a 0-10 scale by the judge/reviewer
        scaled = (raw / (max_possible * 10.0)) * ruleset.freestyle_evaluation_scale_max

    return raw, scaled, warnings


def _confidence(
    events: list[AnalysisEventPrediction], deductions: list[DeductionPrediction]
) -> float:
    confidences = [e.confidence for e in events] + [d.confidence for d in deductions]
    if not confidences:
        return 1.0
    return sum(confidences) / len(confidences)


class DeterministicScoringEngine:
    """Default implementation of the `ScoringEngine` protocol."""

    def calculate(
        self,
        events: list[AnalysisEventPrediction],
        deductions: list[DeductionPrediction],
        freestyle_evaluation: FreestyleEvaluation | None,
        ruleset: Ruleset,
    ) -> ScoreBreakdown:
        warnings: list[str] = [_UNOFFICIAL_WARNING]
        if not ruleset.is_official:
            warnings.append(
                f"Ruleset '{ruleset.version}' is an unofficial draft: {ruleset.disclaimer}"
            )

        technical_raw, technical_warnings = _technical_points(events, ruleset)
        technical_scaled = min(technical_raw, ruleset.technical_scale_max)

        deduction_total, deduction_warnings = _deduction_points(deductions, ruleset)

        fe_raw, fe_scaled, fe_warnings = _freestyle_evaluation_points(freestyle_evaluation, ruleset)

        low_conf_count = sum(
            1 for e in events if e.confidence < ruleset.low_confidence_review_threshold
        ) + sum(1 for d in deductions if d.confidence < ruleset.low_confidence_review_threshold)
        if low_conf_count:
            warnings.append(
                f"{low_conf_count} detection(s) below confidence threshold "
                f"({ruleset.low_confidence_review_threshold:.0%}) require human review."
            )

        final_score = (
            ruleset.technical_weight * technical_scaled
            + ruleset.freestyle_evaluation_weight * fe_scaled
            - deduction_total
        )
        final_score = max(0.0, final_score)

        return ScoreBreakdown(
            technical_raw=round(technical_raw, 3),
            technical_scaled=round(technical_scaled, 3),
            freestyle_evaluation_raw=round(fe_raw, 3),
            freestyle_evaluation_scaled=round(fe_scaled, 3),
            major_deductions=round(deduction_total, 3),
            final_score=round(final_score, 3),
            confidence=round(_confidence(events, deductions), 3),
            ruleset_version=ruleset.version,
            warnings=[*warnings, *technical_warnings, *deduction_warnings, *fe_warnings],
        )


def score_preview_at_ms(
    events: list[AnalysisEventPrediction],
    deductions: list[DeductionPrediction],
    freestyle_evaluation: FreestyleEvaluation | None,
    ruleset: Ruleset,
    up_to_ms: int,
) -> ScoreBreakdown:
    """Scores only tricks completed by `up_to_ms` (`end_ms <= up_to_ms`) and
    deductions that have occurred by then (`timestamp_ms <= up_to_ms`).
    Freestyle Evaluation is not playhead-gated -- it reflects the full manual
    entry for the routine."""
    if up_to_ms < 0:
        raise ValueError("up_to_ms must be non-negative")

    completed_events = [event for event in events if event.end_ms <= up_to_ms]
    occurred_deductions = [
        deduction for deduction in deductions if deduction.timestamp_ms <= up_to_ms
    ]
    return DeterministicScoringEngine().calculate(
        events=completed_events,
        deductions=occurred_deductions,
        freestyle_evaluation=freestyle_evaluation,
        ruleset=ruleset,
    )
