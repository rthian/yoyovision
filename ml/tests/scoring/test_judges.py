"""Prompt D requirements 7-8: "Support manually entered judge clicks" and
"Support multiple human judge scores." Tests `scoring.judges`."""

from __future__ import annotations

from conftest import make_event_prediction, make_judge_click, make_judge_score

from yoyovision_ml.domain import EventFamily, Outcome, Source
from yoyovision_ml.scoring.judges import (
    FE_CATEGORIES,
    aggregate_judge_scores,
    match_clicks_to_events,
    pairwise_judge_agreement,
)


def test_fe_categories_covers_all_eight_including_showmanship() -> None:
    assert FE_CATEGORIES == (
        "execution",
        "control",
        "trick_diversity",
        "space_use_emphasis",
        "music_choreography",
        "music_construction",
        "body_control",
        "showmanship",
    )


def test_aggregate_judge_scores_with_no_scores_returns_all_none_and_warns() -> None:
    evaluation, warnings = aggregate_judge_scores([])
    assert evaluation.execution is None
    assert evaluation.showmanship is None
    assert evaluation.source == Source.HUMAN
    assert len(warnings) == 1
    assert "no" in warnings[0].lower()


def test_aggregate_judge_scores_averages_per_category() -> None:
    scores = [
        make_judge_score(judge_id="judge-a", execution=6.0),
        make_judge_score(judge_id="judge-b", execution=8.0),
    ]
    evaluation, warnings = aggregate_judge_scores(scores)
    assert evaluation.execution == 7.0
    assert "judge-a" in evaluation.notes
    assert "judge-b" in evaluation.notes
    assert not any("disagree" in w for w in warnings)


def test_aggregate_judge_scores_warns_on_missing_category() -> None:
    scores = [make_judge_score(judge_id="judge-a", music_construction=None)]
    _, warnings = aggregate_judge_scores(scores)
    assert any("music_construction" in w for w in warnings)


def test_aggregate_judge_scores_warns_on_sharp_disagreement() -> None:
    scores = [
        make_judge_score(judge_id="judge-a", execution=2.0),
        make_judge_score(judge_id="judge-b", execution=9.0),
    ]
    _, warnings = aggregate_judge_scores(scores)
    assert any("disagree" in w and "execution" in w for w in warnings)


def test_aggregate_judge_scores_only_averages_judges_who_entered_a_category() -> None:
    scores = [
        make_judge_score(judge_id="judge-a", showmanship=10.0),
        make_judge_score(judge_id="judge-b", showmanship=None),
    ]
    evaluation, _ = aggregate_judge_scores(scores)
    assert evaluation.showmanship == 10.0


def test_pairwise_judge_agreement_computes_absolute_difference_per_category() -> None:
    scores = [
        make_judge_score(judge_id="judge-a", execution=6.0, control=5.0),
        make_judge_score(judge_id="judge-b", execution=8.0, control=5.0),
    ]
    results = pairwise_judge_agreement(scores)
    execution_result = next(r for r in results if r.category == "execution")
    control_result = next(r for r in results if r.category == "control")
    assert execution_result.absolute_difference == 2.0
    assert control_result.absolute_difference == 0.0


def test_pairwise_judge_agreement_skips_missing_category_values() -> None:
    scores = [
        make_judge_score(judge_id="judge-a", music_choreography=None),
        make_judge_score(judge_id="judge-b", music_choreography=7.0),
    ]
    results = pairwise_judge_agreement(scores)
    assert all(r.category != "music_choreography" for r in results)


def test_pairwise_judge_agreement_ignores_same_judge_id_pairs() -> None:
    scores = [make_judge_score(judge_id="judge-a"), make_judge_score(judge_id="judge-a")]
    assert pairwise_judge_agreement(scores) == []


def test_pairwise_judge_agreement_empty_or_single_judge_returns_empty() -> None:
    assert pairwise_judge_agreement([]) == []
    assert pairwise_judge_agreement([make_judge_score()]) == []


def test_match_clicks_to_events_finds_closest_within_tolerance() -> None:
    events = [
        make_event_prediction(label="mount_1", start_ms=1000),
        make_event_prediction(label="hop_1", family=EventFamily.HOP, start_ms=5000),
    ]
    clicks = [make_judge_click(timestamp_ms=1050)]

    matches = match_clicks_to_events(clicks, events, tolerance_ms=500)

    assert matches[0].matched_event_label == "mount_1"
    assert matches[0].boundary_error_ms == 1000 - 1050


def test_match_clicks_to_events_no_match_outside_tolerance() -> None:
    events = [make_event_prediction(label="mount_1", start_ms=1000)]
    clicks = [make_judge_click(timestamp_ms=5000)]

    matches = match_clicks_to_events(clicks, events, tolerance_ms=500)

    assert matches[0].matched_event_label is None
    assert matches[0].boundary_error_ms is None


def test_match_clicks_to_events_respects_associated_label_filter() -> None:
    events = [
        make_event_prediction(label="mount_1", start_ms=1000),
        make_event_prediction(label="hop_1", family=EventFamily.HOP, start_ms=1010),
    ]
    clicks = [make_judge_click(timestamp_ms=1000, associated_label="hop_1")]

    matches = match_clicks_to_events(clicks, events, tolerance_ms=500)

    assert matches[0].matched_event_label == "hop_1"


def test_match_clicks_to_events_picks_closest_of_multiple_candidates() -> None:
    events = [
        make_event_prediction(label="mount_1", start_ms=800),
        make_event_prediction(label="mount_1", start_ms=1100),
    ]
    clicks = [make_judge_click(timestamp_ms=1000)]

    matches = match_clicks_to_events(clicks, events, tolerance_ms=500)

    # 1100 is strictly closer to the 1000ms click (distance 100) than 800 is
    # (distance 200), so it must win regardless of list order.
    assert matches[0].matched_event_label == "mount_1"
    assert matches[0].boundary_error_ms == 1100 - 1000


def test_match_clicks_to_events_ignores_outcome_and_still_matches_misses() -> None:
    events = [
        make_event_prediction(label="mount_1", start_ms=1000, outcome=Outcome.MISS),
    ]
    clicks = [make_judge_click(timestamp_ms=1000)]
    matches = match_clicks_to_events(clicks, events)
    assert matches[0].matched_event_label == "mount_1"
