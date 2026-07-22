from __future__ import annotations

from datetime import UTC, datetime

import pytest

from yoyovision_ml.dataset.schema import AnnotationProvenance, TrickEventAnnotation
from yoyovision_ml.domain import (
    AnalysisEventPrediction,
    DifficultyBand,
    EventFamily,
    FeatureSet,
    Outcome,
    Source,
)
from yoyovision_ml.events.metrics import (
    MatchedPair,
    MatchResult,
    boundary_error,
    confidence_calibration,
    confusion_matrix,
    evaluate,
    evaluate_detector,
    event_precision_recall,
    macro_f1,
    match_events,
    outcome_classification_f1,
    per_class_precision_recall,
    temporal_iou_ms,
    temporal_map,
)
from yoyovision_ml.events.types import TrainingSample


def _gt(
    family: EventFamily,
    start_ms: int,
    end_ms: int,
    outcome: Outcome = Outcome.SUCCESS,
    event_id: str = "gt-0",
) -> TrickEventAnnotation:
    return TrickEventAnnotation(
        event_id=event_id,
        label=family.value,
        family=family,
        start_ms=start_ms,
        end_ms=end_ms,
        outcome=outcome,
        provenance=AnnotationProvenance(
            annotator_id="test", source=Source.HUMAN, annotated_at=datetime(2024, 1, 1, tzinfo=UTC)
        ),
    )


def _pred(
    family: EventFamily,
    start_ms: int,
    end_ms: int,
    confidence: float,
    outcome: Outcome = Outcome.SUCCESS,
) -> AnalysisEventPrediction:
    return AnalysisEventPrediction(
        label=family.value,
        family=family,
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=confidence,
        outcome=outcome,
        difficulty_band=DifficultyBand.UNKNOWN,
        model_name="test-model",
        model_version="0",
    )


# --------------------------------------------------------------------------- #
# temporal_iou_ms
# --------------------------------------------------------------------------- #
def test_temporal_iou_ms_identical_spans_is_one() -> None:
    assert temporal_iou_ms(100, 200, 100, 200) == pytest.approx(1.0)


def test_temporal_iou_ms_disjoint_spans_is_zero() -> None:
    assert temporal_iou_ms(0, 100, 200, 300) == 0.0


def test_temporal_iou_ms_partial_overlap_matches_manual_computation() -> None:
    # a=[100,140] (40ms), b=[100,200] (100ms); inter=40, union=40+100-40=100
    assert temporal_iou_ms(100, 140, 100, 200) == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# match_events
# --------------------------------------------------------------------------- #
def test_match_events_perfect_overlap_same_family_matches() -> None:
    gt = _gt(EventFamily.HOP, 100, 200)
    pred = _pred(EventFamily.HOP, 100, 200, confidence=0.9)
    result = match_events([pred], [gt], tiou_threshold=0.5)
    assert len(result.matches) == 1
    assert result.matches[0].tiou == pytest.approx(1.0)
    assert result.false_positives == []
    assert result.false_negatives == []


def test_match_events_never_matches_across_different_families() -> None:
    gt = _gt(EventFamily.HOP, 100, 200)
    pred = _pred(EventFamily.SLACK, 100, 200, confidence=0.9)  # same span, different family
    result = match_events([pred], [gt], tiou_threshold=0.5)
    assert result.matches == []
    assert result.false_positives == [pred]
    assert result.false_negatives == [gt]


def test_match_events_below_threshold_is_a_false_positive_and_false_negative() -> None:
    gt = _gt(EventFamily.HOP, 100, 200)
    pred = _pred(EventFamily.HOP, 100, 120, confidence=0.9)  # tiou well below 0.5
    result = match_events([pred], [gt], tiou_threshold=0.5)
    assert result.matches == []
    assert result.false_positives == [pred]
    assert result.false_negatives == [gt]


def test_match_events_is_greedy_by_confidence_not_by_best_tiou() -> None:
    """A lower-tiou, higher-confidence prediction claims the ground-truth
    event before a later, worse-confidence-but-better-tiou prediction gets a
    chance -- `match_events`' documented "highest-confidence-first" greedy
    matching, not "best tIoU overall"."""
    gt = _gt(EventFamily.HOP, 100, 200)
    high_confidence_worse_tiou = _pred(EventFamily.HOP, 100, 140, confidence=0.9)  # tiou 0.4
    low_confidence_perfect_tiou = _pred(EventFamily.HOP, 100, 200, confidence=0.5)  # tiou 1.0

    result = match_events(
        [high_confidence_worse_tiou, low_confidence_perfect_tiou], [gt], tiou_threshold=0.3
    )

    assert len(result.matches) == 1
    assert result.matches[0].predicted is high_confidence_worse_tiou
    assert result.matches[0].tiou == pytest.approx(0.4)
    assert result.false_positives == [low_confidence_perfect_tiou]
    assert result.false_negatives == []


def test_match_events_matches_the_highest_tiou_gt_among_multiple_candidates() -> None:
    gt_far = _gt(EventFamily.HOP, 500, 600, event_id="gt-far")
    gt_near = _gt(EventFamily.HOP, 100, 200, event_id="gt-near")
    pred = _pred(EventFamily.HOP, 105, 195, confidence=0.9)

    result = match_events([pred], [gt_far, gt_near], tiou_threshold=0.3)

    assert len(result.matches) == 1
    assert result.matches[0].ground_truth is gt_near
    assert result.false_negatives == [gt_far]


# --------------------------------------------------------------------------- #
# event_precision_recall / per_class_precision_recall / macro_f1
# --------------------------------------------------------------------------- #
def test_event_precision_recall_computes_pooled_metrics() -> None:
    gt_hop = _gt(EventFamily.HOP, 0, 100, event_id="gt-hop")
    gt_slack = _gt(EventFamily.SLACK, 200, 300, event_id="gt-slack")
    pred_hop = _pred(EventFamily.HOP, 0, 100, confidence=0.9)  # true positive
    pred_slack_fp = _pred(EventFamily.SLACK, 900, 1000, confidence=0.8)  # false positive

    result = match_events([pred_hop, pred_slack_fp], [gt_hop, gt_slack], tiou_threshold=0.5)
    pr = event_precision_recall(result)

    assert pr.true_positives == 1
    assert pr.false_positives == 1
    assert pr.false_negatives == 1
    assert pr.precision == pytest.approx(0.5)
    assert pr.recall == pytest.approx(0.5)
    assert pr.f1 == pytest.approx(0.5)


def test_per_class_precision_recall_and_macro_f1_split_by_family() -> None:
    gt_hop = _gt(EventFamily.HOP, 0, 100, event_id="gt-hop")
    gt_slack = _gt(EventFamily.SLACK, 200, 300, event_id="gt-slack")
    pred_hop = _pred(EventFamily.HOP, 0, 100, confidence=0.9)  # matches gt_hop perfectly
    pred_slack_fp = _pred(EventFamily.SLACK, 900, 1000, confidence=0.8)  # no matching gt

    per_class = per_class_precision_recall(
        [pred_hop, pred_slack_fp], [gt_hop, gt_slack], tiou_threshold=0.5
    )

    assert per_class["hop"].f1 == pytest.approx(1.0)
    assert per_class["slack"].f1 == pytest.approx(0.0)
    assert macro_f1(per_class) == pytest.approx(0.5)


def test_macro_f1_of_empty_per_class_dict_is_zero() -> None:
    assert macro_f1({}) == 0.0


# --------------------------------------------------------------------------- #
# temporal_map
# --------------------------------------------------------------------------- #
def test_temporal_map_is_one_when_every_prediction_perfectly_matches() -> None:
    gt1 = _gt(EventFamily.HOP, 0, 100, event_id="gt-1")
    gt2 = _gt(EventFamily.HOP, 200, 300, event_id="gt-2")
    pred1 = _pred(EventFamily.HOP, 0, 100, confidence=0.9)
    pred2 = _pred(EventFamily.HOP, 200, 300, confidence=0.8)

    result = temporal_map([pred1, pred2], [gt1, gt2], tiou_thresholds=(0.5,))

    assert result.ap_by_threshold[0.5]["hop"] == pytest.approx(1.0)
    assert result.map_by_threshold[0.5] == pytest.approx(1.0)
    assert result.mean_map == pytest.approx(1.0)


def test_temporal_map_penalizes_a_high_confidence_false_positive() -> None:
    gt1 = _gt(EventFamily.HOP, 0, 100, event_id="gt-1")
    gt2 = _gt(EventFamily.HOP, 200, 300, event_id="gt-2")
    false_positive = _pred(EventFamily.HOP, 500, 600, confidence=0.95)
    pred1 = _pred(EventFamily.HOP, 0, 100, confidence=0.9)
    pred2 = _pred(EventFamily.HOP, 200, 300, confidence=0.8)

    result = temporal_map(
        [false_positive, pred1, pred2], [gt1, gt2], tiou_thresholds=(0.5,)
    )
    assert result.ap_by_threshold[0.5]["hop"] == pytest.approx(0.6667, abs=1e-4)


def test_temporal_map_excludes_families_with_no_ground_truth() -> None:
    gt = _gt(EventFamily.HOP, 0, 100)
    pred_other_family = _pred(EventFamily.SLACK, 0, 100, confidence=0.9)
    result = temporal_map([pred_other_family], [gt], tiou_thresholds=(0.5,))
    assert set(result.ap_by_threshold[0.5]) == {"hop"}


def test_temporal_map_of_no_ground_truth_at_all_is_zero() -> None:
    result = temporal_map([], [], tiou_thresholds=(0.5,))
    assert result.map_by_threshold[0.5] == 0.0
    assert result.mean_map == 0.0


# --------------------------------------------------------------------------- #
# boundary_error
# --------------------------------------------------------------------------- #
def test_boundary_error_computes_mean_absolute_error_over_matches() -> None:
    gt = _gt(EventFamily.HOP, 100, 200)
    pred = _pred(EventFamily.HOP, 105, 195, confidence=0.9)
    result = match_events([pred], [gt], tiou_threshold=0.5)
    error = boundary_error(result)
    assert error.start_mae_ms == pytest.approx(5.0)
    assert error.end_mae_ms == pytest.approx(5.0)
    assert error.matched_count == 1


def test_boundary_error_of_no_matches_is_zero() -> None:
    error = boundary_error(MatchResult())
    assert error.start_mae_ms == 0.0
    assert error.end_mae_ms == 0.0
    assert error.matched_count == 0


# --------------------------------------------------------------------------- #
# outcome_classification_f1
# --------------------------------------------------------------------------- #
def test_outcome_classification_f1_of_no_matches_is_all_zero() -> None:
    result = outcome_classification_f1(MatchResult())
    assert result["success"] == 0.0
    assert result["miss"] == 0.0
    assert result["uncertain"] == 0.0
    assert result["macro"] == 0.0


def test_outcome_classification_f1_scores_correct_and_incorrect_outcomes() -> None:
    gt_success = _gt(EventFamily.HOP, 0, 100, outcome=Outcome.SUCCESS, event_id="gt-1")
    gt_uncertain = _gt(EventFamily.SLACK, 200, 300, outcome=Outcome.UNCERTAIN, event_id="gt-2")
    pred_success = _pred(EventFamily.HOP, 0, 100, confidence=0.9, outcome=Outcome.SUCCESS)
    pred_miss = _pred(EventFamily.SLACK, 200, 300, confidence=0.8, outcome=Outcome.MISS)

    match_result = MatchResult(
        matches=[
            MatchedPair(pred_success, gt_success, 1.0),
            MatchedPair(pred_miss, gt_uncertain, 1.0),
        ]
    )
    result = outcome_classification_f1(match_result)

    assert result["success"] == pytest.approx(1.0)
    assert result["miss"] == pytest.approx(0.0)
    assert result["uncertain"] == pytest.approx(0.0)
    assert result["macro"] == pytest.approx(1.0 / 3.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# confidence_calibration
# --------------------------------------------------------------------------- #
def test_confidence_calibration_reports_zero_error_for_perfectly_calibrated_predictions() -> None:
    gt = _gt(EventFamily.HOP, 0, 100)
    matched_pred = _pred(EventFamily.HOP, 0, 100, confidence=1.0)
    unmatched_pred = _pred(EventFamily.SLACK, 500, 600, confidence=0.0)
    match_result = match_events([matched_pred, unmatched_pred], [gt], tiou_threshold=0.5)

    calibration = confidence_calibration([matched_pred, unmatched_pred], match_result)
    assert calibration["expected_calibration_error"] == pytest.approx(0.0, abs=1e-6)
    assert calibration["brier_score"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# confusion_matrix
# --------------------------------------------------------------------------- #
def test_confusion_matrix_reports_matches_false_positives_and_false_negatives() -> None:
    gt_hop = _gt(EventFamily.HOP, 0, 100, event_id="gt-hop")
    gt_roll = _gt(EventFamily.ROLL, 900, 1000, event_id="gt-roll")
    pred_hop = _pred(EventFamily.HOP, 0, 100, confidence=0.9)
    pred_slack_fp = _pred(EventFamily.SLACK, 500, 600, confidence=0.8)

    result = match_events([pred_hop, pred_slack_fp], [gt_hop, gt_roll], tiou_threshold=0.5)
    matrix = confusion_matrix(result)

    assert matrix["hop"]["hop"] == 1
    assert matrix["slack"]["none"] == 1
    assert matrix["none"]["roll"] == 1


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def test_evaluate_report_to_dict_has_every_prompt_c_metric_key() -> None:
    gt = _gt(EventFamily.HOP, 0, 100)
    pred = _pred(EventFamily.HOP, 0, 100, confidence=0.9)
    report = evaluate([pred], [gt])
    payload = report.to_dict()
    assert set(payload) == {
        "event_precision_recall",
        "per_class_precision_recall",
        "macro_f1",
        "temporal_map",
        "boundary_error",
        "outcome_f1",
        "calibration",
        "confusion_matrix",
    }


def test_evaluate_macro_f1_matches_standalone_macro_f1_helper() -> None:
    gt_hop = _gt(EventFamily.HOP, 0, 100, event_id="gt-hop")
    gt_slack = _gt(EventFamily.SLACK, 200, 300, event_id="gt-slack")
    pred_hop = _pred(EventFamily.HOP, 0, 100, confidence=0.9)
    pred_slack_fp = _pred(EventFamily.SLACK, 900, 1000, confidence=0.8)

    report = evaluate([pred_hop, pred_slack_fp], [gt_hop, gt_slack])
    per_class = per_class_precision_recall([pred_hop, pred_slack_fp], [gt_hop, gt_slack])
    assert report.macro_f1 == macro_f1(per_class)


# --------------------------------------------------------------------------- #
# evaluate_detector (multi-clip offsetting)
# --------------------------------------------------------------------------- #
class _FixedDetector:
    """Ignores whatever `FeatureSet` it is given and always predicts the same
    single event at [0, 100] -- lets the test isolate `evaluate_detector`'s
    per-clip millisecond-offsetting behaviour from any real model logic."""

    def predict(
        self, _features: FeatureSet
    ) -> tuple[list[AnalysisEventPrediction], list[object]]:
        return [_pred(EventFamily.HOP, 0, 100, confidence=0.9)], []


def _sample_with_matching_ground_truth(video_id: str, player_id: str) -> TrainingSample:
    features = FeatureSet(frames=(), feature_names=(), fps=30.0)
    return TrainingSample(
        video_id=video_id,
        player_id=player_id,
        features=features,
        trick_events=(_gt(EventFamily.HOP, 0, 100, event_id=f"{video_id}-gt"),),
    )


def test_evaluate_detector_scores_each_clip_independently_via_offsetting() -> None:
    """Both clips have identical relative-timestamp ground truth/prediction
    (a perfect match); if clip offsetting were missing or wrong, the second
    clip's [0, 100] span could spuriously "already be matched" by the
    first's, silently under-counting true positives."""
    samples = [
        _sample_with_matching_ground_truth("clip-a", "player-a"),
        _sample_with_matching_ground_truth("clip-b", "player-b"),
    ]
    report = evaluate_detector(_FixedDetector(), samples, tiou_threshold=0.5)
    assert report.event_precision_recall.true_positives == 2
    assert report.event_precision_recall.false_positives == 0
    assert report.event_precision_recall.false_negatives == 0
