"""Prompt D requirement 2: "Separate: event detection / technical counting /
technical scaling / Freestyle Evaluation / major deductions" and requirement
10: "Output confidence intervals or uncertainty ranges where supported."
Tests `scoring.pipeline.run_scoring_pipeline`, the orchestrator tying every
other `scoring` submodule together."""

from __future__ import annotations

from conftest import (
    make_analysis_event,
    make_event_override,
    make_judge_score,
    make_major_deduction,
)

from yoyovision_ml.domain import (
    DeductionType,
    DifficultyBand,
    EventFamily,
    FreestyleEvaluation,
    Outcome,
    ReviewStatus,
    Source,
)
from yoyovision_ml.ruleset import default_ruleset
from yoyovision_ml.scoring.pipeline import run_scoring_pipeline
from yoyovision_ml.scoring.profiles import ScoringProfile


def test_empty_input_produces_zero_score_with_no_crash() -> None:
    ruleset = default_ruleset()
    result = run_scoring_pipeline(events=[], deductions=[], ruleset=ruleset)
    assert result.breakdown.final_score == 0.0
    assert result.technical_event_count == 0
    # No human value was ever provided, but `trick_diversity`'s automatic
    # estimator always produces a value (0.0 out of 0 events, never None),
    # so under judge_assist's default `use_automatic_fe_estimators=True`
    # the source is "estimated", not "none" -- "none" is only reachable
    # when automatic estimators are disabled entirely.
    assert result.freestyle_evaluation_source == "estimated"


def test_pending_events_count_under_judge_assist_profile() -> None:
    ruleset = default_ruleset()
    events = [
        make_analysis_event(
            "evt-1", outcome=Outcome.SUCCESS, review_status=ReviewStatus.PENDING
        )
    ]
    result = run_scoring_pipeline(
        events=events, deductions=[], ruleset=ruleset, profile=ScoringProfile.JUDGE_ASSIST
    )
    assert result.technical_event_count == 1
    assert result.technical_raw > 0.0


def test_rejected_events_never_count_regardless_of_profile() -> None:
    ruleset = default_ruleset()
    events = [
        make_analysis_event(
            "evt-1", outcome=Outcome.SUCCESS, review_status=ReviewStatus.REJECTED
        )
    ]
    for profile in ScoringProfile:
        result = run_scoring_pipeline(
            events=events, deductions=[], ruleset=ruleset, profile=profile
        )
        assert result.technical_event_count == 0, f"profile={profile}"


def test_research_profile_excludes_pending_events() -> None:
    ruleset = default_ruleset()
    events = [
        make_analysis_event(
            "evt-1", outcome=Outcome.SUCCESS, review_status=ReviewStatus.PENDING
        )
    ]
    result = run_scoring_pipeline(
        events=events, deductions=[], ruleset=ruleset, profile=ScoringProfile.RESEARCH
    )
    assert result.technical_event_count == 0


def test_research_profile_includes_confirmed_events() -> None:
    ruleset = default_ruleset()
    events = [
        make_analysis_event(
            "evt-1", outcome=Outcome.SUCCESS, review_status=ReviewStatus.CONFIRMED
        )
    ]
    result = run_scoring_pipeline(
        events=events, deductions=[], ruleset=ruleset, profile=ScoringProfile.RESEARCH
    )
    assert result.technical_event_count == 1


def test_dangerous_play_flag_never_reduces_score_until_confirmed() -> None:
    """The single most important Prompt D guarantee, exercised through the
    full orchestrator: a PENDING dangerous_play_review MajorDeduction is
    counted as `deductions_awaiting_confirmation` but contributes zero to
    `major_deductions`; only once a human sets it to CONFIRMED does it
    actually reduce the score."""
    ruleset = default_ruleset()
    pending_deduction = make_major_deduction(
        type_=DeductionType.DANGEROUS_PLAY_REVIEW, review_status=ReviewStatus.PENDING
    )
    pending_result = run_scoring_pipeline(
        events=[], deductions=[pending_deduction], ruleset=ruleset
    )
    assert pending_result.major_deductions == 0.0
    assert pending_result.deductions_awaiting_confirmation == 1

    confirmed_deduction = make_major_deduction(
        type_=DeductionType.DANGEROUS_PLAY_REVIEW, review_status=ReviewStatus.CONFIRMED
    )
    confirmed_result = run_scoring_pipeline(
        events=[], deductions=[confirmed_deduction], ruleset=ruleset
    )
    assert confirmed_result.major_deductions > 0.0
    assert confirmed_result.deductions_awaiting_confirmation == 0


def test_rejected_dangerous_play_flag_is_never_awaiting_confirmation() -> None:
    ruleset = default_ruleset()
    rejected_deduction = make_major_deduction(
        type_=DeductionType.DANGEROUS_PLAY_REVIEW, review_status=ReviewStatus.REJECTED
    )
    result = run_scoring_pipeline(events=[], deductions=[rejected_deduction], ruleset=ruleset)
    assert result.major_deductions == 0.0
    assert result.deductions_awaiting_confirmation == 0


def test_ordinary_pending_deduction_counts_immediately_no_confirmation_needed() -> None:
    ruleset = default_ruleset()
    deduction = make_major_deduction(
        type_=DeductionType.YOYO_STOP, review_status=ReviewStatus.PENDING
    )
    result = run_scoring_pipeline(events=[], deductions=[deduction], ruleset=ruleset)
    assert result.major_deductions > 0.0
    assert result.deductions_awaiting_confirmation == 0


def test_event_overrides_are_applied_and_audit_logged() -> None:
    ruleset = default_ruleset()
    events = [make_analysis_event("evt-1", outcome=Outcome.MISS)]
    overrides = [
        make_event_override(
            event_id="evt-1", field_name="outcome", overridden_value="success"
        )
    ]
    result = run_scoring_pipeline(
        events=events, deductions=[], ruleset=ruleset, event_overrides=overrides
    )
    assert result.technical_raw > 0.0
    assert len(result.override_audit_log) == 1


def test_human_evaluation_is_used_when_no_judge_scores_given() -> None:
    ruleset = default_ruleset()
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
    )
    result = run_scoring_pipeline(
        events=[], deductions=[], ruleset=ruleset, human_evaluation=evaluation
    )
    assert result.freestyle_evaluation_source == "human"
    assert result.freestyle_evaluation_scaled == 80.0


def test_judge_scores_take_precedence_over_human_evaluation_with_warning() -> None:
    ruleset = default_ruleset()
    human_evaluation = FreestyleEvaluation(
        execution=1.0,
        control=1.0,
        trick_diversity=1.0,
        space_use_emphasis=1.0,
        music_choreography=1.0,
        music_construction=1.0,
        body_control=1.0,
        showmanship=1.0,
        source=Source.HUMAN,
    )
    judge_scores = [
        make_judge_score(judge_id="judge-a", showmanship=9.0),
        make_judge_score(judge_id="judge-b", showmanship=9.0),
    ]
    result = run_scoring_pipeline(
        events=[],
        deductions=[],
        ruleset=ruleset,
        human_evaluation=human_evaluation,
        judge_scores=judge_scores,
    )
    assert any("ignoring human_evaluation" in w for w in result.warnings)
    # human_evaluation (all 1.0 -> 10.0 scaled) must have been ignored, not blended in.
    assert result.breakdown.freestyle_evaluation_scaled != 10.0


def test_automatic_fe_estimators_fill_gaps_when_judge_scores_incomplete() -> None:
    ruleset = default_ruleset()
    events = [
        make_analysis_event("evt-1", outcome=Outcome.SUCCESS),
    ]
    # A single judge who only entered showmanship -- every estimatable
    # category is left None and should be auto-filled.
    judge_scores = [
        make_judge_score(
            judge_id="judge-a",
            execution=None,
            control=None,
            trick_diversity=None,
            space_use_emphasis=None,
            music_choreography=None,
            music_construction=None,
            body_control=None,
            showmanship=9.0,
        )
    ]
    result = run_scoring_pipeline(
        events=events, deductions=[], ruleset=ruleset, judge_scores=judge_scores
    )
    assert result.freestyle_evaluation_source == "human+estimated"
    assert len(result.fe_estimates) > 0


def test_practice_profile_gives_repeated_high_risk_elements_full_credit() -> None:
    ruleset = default_ruleset()
    events = [
        make_analysis_event(
            "evt-1",
            label="suicide_1",
            family=EventFamily.SUICIDE,
            band=DifficultyBand.ADVANCED,
            outcome=Outcome.SUCCESS,
        ),
        make_analysis_event(
            "evt-2",
            label="suicide_1",
            family=EventFamily.SUICIDE,
            band=DifficultyBand.ADVANCED,
            outcome=Outcome.SUCCESS,
            start_ms=1000,
        ),
    ]
    result = run_scoring_pipeline(
        events=events, deductions=[], ruleset=ruleset, profile=ScoringProfile.PRACTICE
    )
    assert result.technical_raw == round(ruleset.difficulty_band_points.advanced * 2, 3)


def test_practice_profile_does_not_mutate_the_shared_ruleset_object() -> None:
    ruleset = default_ruleset()
    original_policy = ruleset.repeated_element_decay.policy
    run_scoring_pipeline(
        events=[], deductions=[], ruleset=ruleset, profile=ScoringProfile.PRACTICE
    )
    assert ruleset.repeated_element_decay.policy == original_policy


def test_practice_profile_disables_confidence_interval() -> None:
    ruleset = default_ruleset()
    events = [make_analysis_event("evt-1", outcome=Outcome.SUCCESS)]
    result = run_scoring_pipeline(
        events=events, deductions=[], ruleset=ruleset, profile=ScoringProfile.PRACTICE
    )
    assert result.final_score_interval is None


def test_judge_assist_profile_computes_confidence_interval_with_enough_events() -> None:
    ruleset = default_ruleset()
    events = [
        make_analysis_event(f"evt-{i}", label=f"mount_{i}", start_ms=i * 1000)
        for i in range(5)
    ]
    result = run_scoring_pipeline(
        events=events,
        deductions=[],
        ruleset=ruleset,
        profile=ScoringProfile.JUDGE_ASSIST,
        bootstrap_iterations=50,
        bootstrap_seed=1,
    )
    assert result.final_score_interval is not None
    lower, upper = result.final_score_interval
    assert lower <= upper
    assert lower >= 0.0


def test_confidence_interval_reproducible_with_same_seed() -> None:
    ruleset = default_ruleset()
    events = [
        make_analysis_event(f"evt-{i}", label=f"mount_{i}", start_ms=i * 1000)
        for i in range(5)
    ]
    result_a = run_scoring_pipeline(
        events=events,
        deductions=[],
        ruleset=ruleset,
        bootstrap_iterations=50,
        bootstrap_seed=42,
    )
    result_b = run_scoring_pipeline(
        events=events,
        deductions=[],
        ruleset=ruleset,
        bootstrap_iterations=50,
        bootstrap_seed=42,
    )
    assert result_a.final_score_interval == result_b.final_score_interval


def test_no_confidence_interval_warning_when_nothing_to_resample() -> None:
    ruleset = default_ruleset()
    result = run_scoring_pipeline(
        events=[], deductions=[], ruleset=ruleset, profile=ScoringProfile.JUDGE_ASSIST
    )
    assert result.final_score_interval is None
    assert any("bootstrap confidence interval" in w for w in result.warnings)


def test_research_profile_warns_when_fewer_than_two_judges_supplied() -> None:
    ruleset = default_ruleset()
    result = run_scoring_pipeline(
        events=[],
        deductions=[],
        ruleset=ruleset,
        profile=ScoringProfile.RESEARCH,
        judge_scores=[make_judge_score(judge_id="judge-a")],
    )
    assert any("expects multiple judge scores" in w for w in result.warnings)


def test_result_carries_ruleset_and_profile_identity_for_audit() -> None:
    ruleset = default_ruleset()
    result = run_scoring_pipeline(
        events=[], deductions=[], ruleset=ruleset, profile=ScoringProfile.RESEARCH
    )
    assert result.ruleset_version == ruleset.version
    assert result.profile == "research"


def test_breakdown_warnings_are_included_in_result_warnings() -> None:
    ruleset = default_ruleset()
    result = run_scoring_pipeline(events=[], deductions=[], ruleset=ruleset)
    assert any("unofficial" in w.lower() for w in result.warnings)
