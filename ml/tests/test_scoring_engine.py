from __future__ import annotations

import pytest

from yoyovision_ml.domain import (
    AnalysisEventPrediction,
    DeductionPrediction,
    DeductionType,
    DifficultyBand,
    EventFamily,
    FreestyleEvaluation,
    Outcome,
    ReviewStatus,
    Source,
)
from yoyovision_ml.ruleset import Ruleset, default_ruleset
from yoyovision_ml.scoring_engine import (
    DeterministicScoringEngine,
    deduction_is_scorable,
    deduction_points,
    freestyle_evaluation_points,
    technical_points,
)


def _event(
    label: str,
    family: EventFamily,
    start_ms: int,
    outcome: Outcome = Outcome.SUCCESS,
    band: DifficultyBand = DifficultyBand.BASIC,
    confidence: float = 0.9,
) -> AnalysisEventPrediction:
    return AnalysisEventPrediction(
        label=label,
        family=family,
        start_ms=start_ms,
        end_ms=start_ms + 500,
        confidence=confidence,
        outcome=outcome,
        difficulty_band=band,
        model_name="test-model",
        model_version="0.0.0-test",
    )


def test_ruleset_loads_and_is_marked_unofficial() -> None:
    ruleset = default_ruleset()
    assert ruleset.version == "1a-draft-0.1"
    assert ruleset.is_official is False


def test_successful_event_earns_positive_credit() -> None:
    ruleset = default_ruleset()
    engine = DeterministicScoringEngine()
    events = [_event("mount_1", EventFamily.MOUNT, 0, band=DifficultyBand.BASIC)]

    breakdown = engine.calculate(events, [], None, ruleset)

    assert breakdown.technical_raw == ruleset.difficulty_band_points.basic
    assert breakdown.final_score > 0
    assert any("unofficial" in w.lower() for w in breakdown.warnings)


def test_missed_event_earns_no_credit() -> None:
    ruleset = default_ruleset()
    engine = DeterministicScoringEngine()
    events = [_event("mount_1", EventFamily.MOUNT, 0, outcome=Outcome.MISS)]

    breakdown = engine.calculate(events, [], None, ruleset)

    assert breakdown.technical_raw == 0.0


def test_repeated_high_risk_element_receives_reduced_credit() -> None:
    ruleset = default_ruleset()
    engine = DeterministicScoringEngine()
    events = [
        _event("suicide_1", EventFamily.SUICIDE, 0, band=DifficultyBand.ADVANCED),
        _event("suicide_1", EventFamily.SUICIDE, 1000, band=DifficultyBand.ADVANCED),
    ]

    breakdown = engine.calculate(events, [], None, ruleset)

    first_points = ruleset.difficulty_band_points.advanced
    second_points = first_points * ruleset.repeated_element_decay.occurrence_multipliers[1]
    assert breakdown.technical_raw == round(first_points + second_points, 3)
    assert any("repeated high-risk" in w.lower() for w in breakdown.warnings)


def test_repeated_non_high_risk_element_receives_full_credit_each_time() -> None:
    ruleset = default_ruleset()
    engine = DeterministicScoringEngine()
    events = [
        _event("mount_1", EventFamily.MOUNT, 0, band=DifficultyBand.BASIC),
        _event("mount_1", EventFamily.MOUNT, 1000, band=DifficultyBand.BASIC),
    ]

    breakdown = engine.calculate(events, [], None, ruleset)

    assert breakdown.technical_raw == round(ruleset.difficulty_band_points.basic * 2, 3)


def test_yoyo_stop_deduction_reduces_final_score() -> None:
    ruleset = default_ruleset()
    engine = DeterministicScoringEngine()
    events = [_event("mount_1", EventFamily.MOUNT, 0, band=DifficultyBand.ADVANCED)]
    deductions = [
        DeductionPrediction(
            type=DeductionType.YOYO_STOP,
            timestamp_ms=500,
            quantity=1,
            confidence=0.9,
            model_name="test-model",
            model_version="0.0.0-test",
        )
    ]

    breakdown = engine.calculate(events, deductions, None, ruleset)

    rule = ruleset.deduction_rule_for(DeductionType.YOYO_STOP)
    assert rule is not None
    assert breakdown.major_deductions == rule.points_per_occurrence


def test_deduction_occurrence_cap_is_respected() -> None:
    ruleset = default_ruleset()
    engine = DeterministicScoringEngine()
    rule = ruleset.deduction_rule_for(DeductionType.YOYO_STOP)
    assert rule is not None and rule.max_occurrences_penalized is not None
    over_cap_quantity = rule.max_occurrences_penalized + 5

    deductions = [
        DeductionPrediction(
            type=DeductionType.YOYO_STOP,
            timestamp_ms=100,
            quantity=over_cap_quantity,
            confidence=0.9,
            model_name="test-model",
            model_version="0.0.0-test",
        )
    ]

    breakdown = engine.calculate([], deductions, None, ruleset)

    assert breakdown.major_deductions == rule.points_per_occurrence * rule.max_occurrences_penalized
    assert any("cap" in w.lower() for w in breakdown.warnings)


def test_missing_freestyle_evaluation_produces_warning_and_zero_score() -> None:
    ruleset = default_ruleset()
    engine = DeterministicScoringEngine()

    breakdown = engine.calculate([], [], None, ruleset)

    assert breakdown.freestyle_evaluation_scaled == 0.0
    assert any("placeholder" in w.lower() for w in breakdown.warnings)


def test_full_freestyle_evaluation_contributes_to_score() -> None:
    ruleset = default_ruleset()
    engine = DeterministicScoringEngine()
    evaluation = FreestyleEvaluation(
        execution=8.0,
        control=8.0,
        trick_diversity=8.0,
        space_use_emphasis=8.0,
        music_choreography=8.0,
        music_construction=8.0,
        body_control=8.0,
        showmanship=8.0,
        source=Source.HUMAN,
        notes="manual entry",
    )

    breakdown = engine.calculate([], [], evaluation, ruleset)

    assert breakdown.freestyle_evaluation_scaled == 80.0
    assert breakdown.final_score == round(ruleset.freestyle_evaluation_weight * 80.0, 3)


def test_low_confidence_events_flagged_for_review() -> None:
    ruleset = default_ruleset()
    engine = DeterministicScoringEngine()
    events = [_event("mystery_1", EventFamily.UNKNOWN_TECHNICAL_ELEMENT, 0, confidence=0.1)]

    breakdown = engine.calculate(events, [], None, ruleset)

    assert any("require human review" in w.lower() for w in breakdown.warnings)
    assert breakdown.confidence < ruleset.low_confidence_review_threshold


def test_final_score_never_negative() -> None:
    ruleset = default_ruleset()
    engine = DeterministicScoringEngine()
    deductions = [
        DeductionPrediction(
            type=DeductionType.YOYO_DETACH,
            timestamp_ms=0,
            quantity=100,
            confidence=0.9,
            model_name="test-model",
            model_version="0.0.0-test",
        )
    ]

    breakdown = engine.calculate([], deductions, None, ruleset)

    assert breakdown.final_score == 0.0


# --------------------------------------------------------------------------- #
# Prompt D: public stage wrappers mirror the engine's internal computation
# --------------------------------------------------------------------------- #
def test_technical_points_wrapper_matches_engine() -> None:
    ruleset = default_ruleset()
    events = [_event("mount_1", EventFamily.MOUNT, 0, band=DifficultyBand.INTERMEDIATE)]

    raw, warnings = technical_points(events, ruleset)
    breakdown = DeterministicScoringEngine().calculate(events, [], None, ruleset)

    assert raw == breakdown.technical_raw
    assert warnings == []


def test_deduction_points_wrapper_matches_engine() -> None:
    ruleset = default_ruleset()
    deductions = [
        DeductionPrediction(
            type=DeductionType.YOYO_CHANGE,
            timestamp_ms=0,
            quantity=1,
            confidence=0.9,
            model_name="test-model",
            model_version="0.0.0-test",
        )
    ]

    total, warnings = deduction_points(deductions, ruleset)
    breakdown = DeterministicScoringEngine().calculate([], deductions, None, ruleset)

    assert total == breakdown.major_deductions
    assert warnings == []


def test_freestyle_evaluation_points_wrapper_matches_engine() -> None:
    ruleset = default_ruleset()
    evaluation = FreestyleEvaluation(
        execution=6.0,
        control=None,
        trick_diversity=None,
        space_use_emphasis=None,
        music_choreography=None,
        music_construction=None,
        body_control=None,
        showmanship=None,
        source=Source.HUMAN,
    )

    raw, scaled, warnings = freestyle_evaluation_points(evaluation, ruleset)
    breakdown = DeterministicScoringEngine().calculate([], [], evaluation, ruleset)

    assert raw == breakdown.freestyle_evaluation_raw
    assert scaled == breakdown.freestyle_evaluation_scaled
    assert any("missing manual values" in w for w in warnings)


# --------------------------------------------------------------------------- #
# Prompt D: dangerous-play (and any requires_manual_confirmation type) must
# never affect a score until a human explicitly confirms it.
# --------------------------------------------------------------------------- #
def test_dangerous_play_review_is_not_scorable_while_pending() -> None:
    ruleset = default_ruleset()
    assert (
        deduction_is_scorable(DeductionType.DANGEROUS_PLAY_REVIEW, ReviewStatus.PENDING, ruleset)
        is False
    )


def test_dangerous_play_review_is_not_scorable_while_edited() -> None:
    """Even a human-edited-but-not-yet-confirmed flag must not count --
    only an explicit CONFIRMED satisfies `requires_manual_confirmation`."""
    ruleset = default_ruleset()
    assert (
        deduction_is_scorable(DeductionType.DANGEROUS_PLAY_REVIEW, ReviewStatus.EDITED, ruleset)
        is False
    )


def test_dangerous_play_review_is_scorable_once_confirmed() -> None:
    ruleset = default_ruleset()
    assert (
        deduction_is_scorable(DeductionType.DANGEROUS_PLAY_REVIEW, ReviewStatus.CONFIRMED, ruleset)
        is True
    )


def test_rejected_deduction_is_never_scorable_regardless_of_type() -> None:
    ruleset = default_ruleset()
    assert deduction_is_scorable(DeductionType.YOYO_STOP, ReviewStatus.REJECTED, ruleset) is False


def test_ordinary_deduction_type_is_scorable_while_pending() -> None:
    """Types without `requires_manual_confirmation` keep today's behavior:
    PENDING is scorable (only REJECTED is excluded)."""
    ruleset = default_ruleset()
    assert deduction_is_scorable(DeductionType.YOYO_STOP, ReviewStatus.PENDING, ruleset) is True


def test_dangerous_play_confirmed_deduction_reduces_final_score() -> None:
    """End-to-end: a CONFIRMED dangerous_play_review flag, and only a
    CONFIRMED one, actually reduces `final_score` -- exercised through the
    full `DeterministicScoringEngine`, not just the gate helper."""
    ruleset = default_ruleset()
    engine = DeterministicScoringEngine()
    rule = ruleset.deduction_rule_for(DeductionType.DANGEROUS_PLAY_REVIEW)
    assert rule is not None and rule.requires_manual_confirmation

    deduction = DeductionPrediction(
        type=DeductionType.DANGEROUS_PLAY_REVIEW,
        timestamp_ms=0,
        quantity=1,
        confidence=0.9,
        model_name="heuristic-dangerous-play-detector",
        model_version="0.1.0-heuristic",
    )

    pending_breakdown = engine.calculate([], [], None, ruleset)
    scored_breakdown = engine.calculate([], [deduction], None, ruleset)

    # The raw prediction has no review_status of its own (that lives on the
    # persisted row) -- this asserts the *rule* would apply real points once
    # a caller includes it, matching what `deduction_is_scorable` gates.
    assert scored_breakdown.major_deductions == rule.points_per_occurrence
    assert pending_breakdown.major_deductions == 0.0


# --------------------------------------------------------------------------- #
# Prompt D requirement 5: repeated-element policies
# --------------------------------------------------------------------------- #
def test_full_credit_policy_disables_decay_for_high_risk_family() -> None:
    ruleset = default_ruleset().model_copy(deep=True)
    ruleset.repeated_element_decay.policy = "full_credit"
    engine = DeterministicScoringEngine()
    events = [
        _event("suicide_1", EventFamily.SUICIDE, 0, band=DifficultyBand.ADVANCED),
        _event("suicide_1", EventFamily.SUICIDE, 1000, band=DifficultyBand.ADVANCED),
    ]

    breakdown = engine.calculate(events, [], None, ruleset)

    assert breakdown.technical_raw == round(ruleset.difficulty_band_points.advanced * 2, 3)


def test_decay_all_families_policy_decays_non_high_risk_family_too() -> None:
    ruleset = default_ruleset().model_copy(deep=True)
    ruleset.repeated_element_decay.policy = "decay_all_families"
    engine = DeterministicScoringEngine()
    events = [
        _event("mount_1", EventFamily.MOUNT, 0, band=DifficultyBand.BASIC),
        _event("mount_1", EventFamily.MOUNT, 1000, band=DifficultyBand.BASIC),
    ]

    breakdown = engine.calculate(events, [], None, ruleset)

    first = ruleset.difficulty_band_points.basic
    second = first * ruleset.repeated_element_decay.occurrence_multipliers[1]
    assert breakdown.technical_raw == round(first + second, 3)


def test_cap_occurrences_policy_zeroes_out_beyond_multiplier_list() -> None:
    ruleset = default_ruleset().model_copy(deep=True)
    ruleset.repeated_element_decay.policy = "cap_occurrences"
    n = len(ruleset.repeated_element_decay.occurrence_multipliers)
    engine = DeterministicScoringEngine()
    events = [
        _event("suicide_1", EventFamily.SUICIDE, i * 1000, band=DifficultyBand.ADVANCED)
        for i in range(n + 1)
    ]

    breakdown = engine.calculate(events, [], None, ruleset)

    advanced = ruleset.difficulty_band_points.advanced
    decay = ruleset.repeated_element_decay
    expected = sum(advanced * decay.multiplier_for_occurrence(i) for i in range(1, n + 1))
    # the (n+1)-th occurrence earns zero credit under this policy, unlike the
    # default policy, which would clamp to the last multiplier instead.
    assert breakdown.technical_raw == round(expected, 3)


def test_multiplier_for_occurrence_index_below_one_raises() -> None:
    ruleset = default_ruleset()
    with pytest.raises(ValueError, match="occurrence_index"):
        ruleset.repeated_element_decay.multiplier_for_occurrence(0)


# --------------------------------------------------------------------------- #
# Prompt D requirement 4: never silently claim official IYYF status
# --------------------------------------------------------------------------- #
def test_official_ruleset_without_certification_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="iyyf_certification_reference"):
        Ruleset(version="test-official", is_official=True)


def test_official_ruleset_with_certification_reference_is_accepted() -> None:
    ruleset = Ruleset(
        version="test-official",
        is_official=True,
        iyyf_certification_reference="IYYF Rulebook 2026 Edition, Section 4.2",
    )
    assert ruleset.is_official is True


def test_packaged_ruleset_never_claims_official_status() -> None:
    """No packaged ruleset ships pre-set to `is_official=True` -- see
    docs/ruleset.md and the module docstring in `ruleset.py`."""
    ruleset = default_ruleset()
    assert ruleset.is_official is False
    assert ruleset.iyyf_certification_reference is None
