"""Prompt D requirements 6-8 domain types: `JudgeClick`, `JudgeFreestyleScore`,
`EventOverride`, `FreestyleEvaluationEstimate`, `ScoringPipelineResult`.
Tests `scoring.types`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from conftest import make_event_override, make_judge_click, make_judge_score

from yoyovision_ml.domain import ScoreBreakdown
from yoyovision_ml.scoring.types import (
    OVERRIDABLE_EVENT_FIELDS,
    FreestyleEvaluationEstimate,
    ScoringPipelineResult,
)


def test_overridable_event_fields_matches_analysis_event_scoring_inputs() -> None:
    assert {
        "label",
        "family",
        "start_ms",
        "end_ms",
        "outcome",
        "difficulty_band",
        "confidence",
    } == OVERRIDABLE_EVENT_FIELDS


def test_judge_click_is_immutable() -> None:
    click = make_judge_click()
    with pytest.raises(FrozenInstanceError):
        click.timestamp_ms = 999  # type: ignore[misc]


def test_judge_freestyle_score_defaults_to_all_none() -> None:
    score = make_judge_score(
        execution=None,
        control=None,
        trick_diversity=None,
        space_use_emphasis=None,
        music_choreography=None,
        music_construction=None,
        body_control=None,
        showmanship=None,
    )
    assert score.execution is None
    assert score.showmanship is None
    assert score.judge_id == "judge-a"


def test_event_override_is_immutable_and_carries_audit_fields() -> None:
    override = make_event_override(reason="manual re-review")
    assert override.reason == "manual re-review"
    assert override.overridden_by == "reviewer-1"
    with pytest.raises(FrozenInstanceError):
        override.overridden_value = "miss"  # type: ignore[misc]


def test_freestyle_evaluation_estimate_allows_none_value_for_declined_guess() -> None:
    """Prompt D: "value=None means the estimator deliberately declined to
    guess ... never a fabricated number." -- exercised directly on the type,
    complementing `test_fe_estimators.py`'s behavioral coverage."""
    estimate = FreestyleEvaluationEstimate(
        category="music_choreography",
        value=None,
        confidence=0.0,
        supporting_features={},
        model_name="heuristic-fe-estimator",
        model_version="0.1.0-heuristic",
        warning="no audio-analysis stage",
    )
    assert estimate.value is None
    assert estimate.confidence == 0.0


def test_scoring_pipeline_result_defaults_warnings_to_empty_list_not_shared() -> None:
    """Guards against a shared mutable-default footgun on the dataclass
    field: two independently constructed results must not share one list."""
    breakdown = ScoreBreakdown(
        technical_raw=0.0,
        technical_scaled=0.0,
        freestyle_evaluation_raw=0.0,
        freestyle_evaluation_scaled=0.0,
        major_deductions=0.0,
        final_score=0.0,
        confidence=1.0,
        ruleset_version="test",
    )
    kwargs = dict(
        profile="judge_assist",
        ruleset_version="test",
        technical_event_count=0,
        technical_raw=0.0,
        technical_scaled=0.0,
        deduction_count=0,
        deductions_awaiting_confirmation=0,
        major_deductions=0.0,
        freestyle_evaluation_raw=0.0,
        freestyle_evaluation_scaled=0.0,
        freestyle_evaluation_source="none",
        fe_estimates=(),
        override_audit_log=(),
        breakdown=breakdown,
    )
    result_a = ScoringPipelineResult(**kwargs)
    result_b = ScoringPipelineResult(**kwargs)
    result_a.warnings.append("only on a")
    assert result_b.warnings == []
