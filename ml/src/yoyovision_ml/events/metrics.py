"""Evaluation metrics for the temporal trick-event model, per Prompt C's
METRICS section: event precision/recall, macro F1, per-class F1, temporal
mAP at configurable tIoU thresholds, start/end boundary error, outcome
classification F1, confidence calibration, and a confusion matrix.

All functions assume every `AnalysisEventPrediction`/`TrickEventAnnotation` passed in
belongs to the same clip/timeline (or has already been offset so cross-clip
timestamps cannot spuriously overlap) -- `evaluate()` is the multi-clip
aggregator callers should use for a whole-dataset report (see `cli.py`'s
`evaluate` command), since match/AP counts are additive across clips while
raw millisecond ranges are not comparable across different clips.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from yoyovision_ml.dataset.schema import TrickEventAnnotation
from yoyovision_ml.domain import AnalysisEventPrediction
from yoyovision_ml.events.calibration import brier_score, expected_calibration_error
from yoyovision_ml.events.labels import EVENT_CLASSES, OUTCOME_CLASSES
from yoyovision_ml.events.types import TrainingSample

#: Millisecond offset applied between clips when pooling multi-clip
#: predictions/ground-truth into one `evaluate()` call (`evaluate_detector`),
#: so distinct clips can never spuriously overlap. 1e9 ms (~11.6 days)
#: comfortably exceeds any real or synthetic clip duration used by this
#: package.
_CLIP_OFFSET_MS = 1_000_000_000


def temporal_iou_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


@dataclass(slots=True, frozen=True)
class MatchedPair:
    predicted: AnalysisEventPrediction
    ground_truth: TrickEventAnnotation
    tiou: float


@dataclass(slots=True, frozen=True)
class MatchResult:
    matches: list[MatchedPair] = field(default_factory=list)
    false_positives: list[AnalysisEventPrediction] = field(default_factory=list)
    false_negatives: list[TrickEventAnnotation] = field(default_factory=list)


def match_events(
    predictions: list[AnalysisEventPrediction],
    ground_truth: list[TrickEventAnnotation],
    tiou_threshold: float = 0.5,
) -> MatchResult:
    """Greedy same-family matching: within each `family`, predictions are
    considered highest-confidence-first; each is matched to the
    highest-tIoU still-unmatched ground-truth event of the same family, if
    that tIoU is `>= tiou_threshold` -- otherwise it is a false positive. Any
    ground-truth event never matched is a false negative."""
    predictions_by_family: dict[str, list[AnalysisEventPrediction]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_family[prediction.family.value].append(prediction)
    ground_truth_by_family: dict[str, list[TrickEventAnnotation]] = defaultdict(list)
    for gt in ground_truth:
        ground_truth_by_family[gt.family.value].append(gt)

    matches: list[MatchedPair] = []
    false_positives: list[AnalysisEventPrediction] = []
    matched_gt_ids: set[str] = set()

    families = set(predictions_by_family) | set(ground_truth_by_family)
    for family in families:
        family_predictions = sorted(
            predictions_by_family.get(family, []), key=lambda p: -p.confidence
        )
        family_gt = ground_truth_by_family.get(family, [])
        matched_in_family: set[str] = set()

        for prediction in family_predictions:
            best_gt: TrickEventAnnotation | None = None
            best_tiou = 0.0
            for gt in family_gt:
                if gt.event_id in matched_in_family:
                    continue
                tiou = temporal_iou_ms(
                    prediction.start_ms, prediction.end_ms, gt.start_ms, gt.end_ms
                )
                if tiou > best_tiou:
                    best_tiou = tiou
                    best_gt = gt
            if best_gt is not None and best_tiou >= tiou_threshold:
                matches.append(MatchedPair(prediction, best_gt, best_tiou))
                matched_in_family.add(best_gt.event_id)
                matched_gt_ids.add(best_gt.event_id)
            else:
                false_positives.append(prediction)

    false_negatives = [gt for gt in ground_truth if gt.event_id not in matched_gt_ids]
    return MatchResult(
        matches=matches, false_positives=false_positives, false_negatives=false_negatives
    )


@dataclass(slots=True, frozen=True)
class PrecisionRecallResult:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def _precision_recall_f1(
    true_positives: int, false_positives: int, false_negatives: int
) -> PrecisionRecallResult:
    denom_p = true_positives + false_positives
    denom_r = true_positives + false_negatives
    precision = true_positives / denom_p if denom_p else 0.0
    recall = true_positives / denom_r if denom_r else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PrecisionRecallResult(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def event_precision_recall(match_result: MatchResult) -> PrecisionRecallResult:
    """Pooled (all-classes-together) event precision/recall/F1."""
    return _precision_recall_f1(
        len(match_result.matches),
        len(match_result.false_positives),
        len(match_result.false_negatives),
    )


def per_class_precision_recall(
    predictions: list[AnalysisEventPrediction],
    ground_truth: list[TrickEventAnnotation],
    tiou_threshold: float = 0.5,
) -> dict[str, PrecisionRecallResult]:
    """One `PrecisionRecallResult` per class that has at least one prediction
    or ground-truth event; classes with neither are omitted (undefined, not
    zero -- see `macro_f1`)."""
    match_result = match_events(predictions, ground_truth, tiou_threshold)
    tp_by_class: dict[str, int] = defaultdict(int)
    for match in match_result.matches:
        tp_by_class[match.predicted.family.value] += 1
    fp_by_class: dict[str, int] = defaultdict(int)
    for fp in match_result.false_positives:
        fp_by_class[fp.family.value] += 1
    fn_by_class: dict[str, int] = defaultdict(int)
    for fn in match_result.false_negatives:
        fn_by_class[fn.family.value] += 1

    classes = {p.family.value for p in predictions} | {g.family.value for g in ground_truth}
    return {
        family: _precision_recall_f1(tp_by_class[family], fp_by_class[family], fn_by_class[family])
        for family in classes
    }


def macro_f1(per_class: dict[str, PrecisionRecallResult]) -> float:
    """Mean F1 across classes with support (at least one prediction or
    ground-truth event) -- classes never seen at all are excluded rather
    than penalized as F1=0, since "no support" is undefined, not "wrong"."""
    if not per_class:
        return 0.0
    return round(sum(result.f1 for result in per_class.values()) / len(per_class), 4)


def _average_precision(
    confidences: list[float], is_true_positive: list[bool], num_ground_truth: int
) -> float:
    """Standard all-points-interpolated AP (area under the precision-recall
    curve), given a confidence-sorted TP/FP sequence for one class."""
    if num_ground_truth == 0 or not confidences:
        return 0.0
    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []
    for i in order:
        if is_true_positive[i]:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / num_ground_truth)

    # All-points interpolation: precision envelope is the running max from
    # the right, integrated over the recall axis.
    precisions_arr = np.array(precisions)
    recalls_arr = np.array(recalls)
    for i in range(len(precisions_arr) - 2, -1, -1):
        precisions_arr[i] = max(precisions_arr[i], precisions_arr[i + 1])

    recall_edges = np.concatenate(([0.0], recalls_arr))
    precision_edges = np.concatenate(([precisions_arr[0]], precisions_arr))
    ap = float(np.sum(np.diff(recall_edges) * precision_edges[1:]))
    return ap


@dataclass(slots=True, frozen=True)
class TemporalMapResult:
    #: `{tiou_threshold: {family: AP}}`
    ap_by_threshold: dict[float, dict[str, float]]
    #: `{tiou_threshold: mAP}` -- mean AP across classes with ground truth, per threshold.
    map_by_threshold: dict[float, float]
    #: Mean of `map_by_threshold` across every threshold.
    mean_map: float


def temporal_map(
    predictions: list[AnalysisEventPrediction],
    ground_truth: list[TrickEventAnnotation],
    tiou_thresholds: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> TemporalMapResult:
    """Temporal mean Average Precision at each of `tiou_thresholds`, computed
    per class then macro-averaged -- Prompt C's "temporal mAP at
    configurable tIoU thresholds"."""
    ap_by_threshold: dict[float, dict[str, float]] = {}
    map_by_threshold: dict[float, float] = {}

    for threshold in tiou_thresholds:
        per_family_ap: dict[str, float] = {}
        for family in {family.value for family in EVENT_CLASSES}:
            family_predictions = [p for p in predictions if p.family.value == family]
            family_gt = [g for g in ground_truth if g.family.value == family]
            if not family_gt:
                continue
            match_result = match_events(family_predictions, family_gt, threshold)
            matched_confidences = [m.predicted.confidence for m in match_result.matches]
            fp_confidences = [fp.confidence for fp in match_result.false_positives]
            confidences = matched_confidences + fp_confidences
            is_tp = [True] * len(matched_confidences) + [False] * len(fp_confidences)
            per_family_ap[family] = round(_average_precision(confidences, is_tp, len(family_gt)), 4)
        ap_by_threshold[threshold] = per_family_ap
        map_by_threshold[threshold] = (
            round(sum(per_family_ap.values()) / len(per_family_ap), 4) if per_family_ap else 0.0
        )

    mean_map = (
        round(sum(map_by_threshold.values()) / len(map_by_threshold), 4)
        if map_by_threshold
        else 0.0
    )
    return TemporalMapResult(
        ap_by_threshold=ap_by_threshold, map_by_threshold=map_by_threshold, mean_map=mean_map
    )


@dataclass(slots=True, frozen=True)
class BoundaryErrorResult:
    start_mae_ms: float
    end_mae_ms: float
    matched_count: int


def boundary_error(match_result: MatchResult) -> BoundaryErrorResult:
    """Mean absolute error (ms) between matched predicted/ground-truth
    start and end times, over true positives only."""
    if not match_result.matches:
        return BoundaryErrorResult(start_mae_ms=0.0, end_mae_ms=0.0, matched_count=0)
    start_errors = [
        abs(m.predicted.start_ms - m.ground_truth.start_ms) for m in match_result.matches
    ]
    end_errors = [abs(m.predicted.end_ms - m.ground_truth.end_ms) for m in match_result.matches]
    return BoundaryErrorResult(
        start_mae_ms=round(sum(start_errors) / len(start_errors), 2),
        end_mae_ms=round(sum(end_errors) / len(end_errors), 2),
        matched_count=len(match_result.matches),
    )


def outcome_classification_f1(match_result: MatchResult) -> dict[str, float]:
    """Per-outcome and macro F1 comparing predicted vs. ground-truth
    `Outcome`, computed only over matched (true-positive) pairs -- an
    unmatched prediction/ground-truth event has no outcome comparison to make."""
    if not match_result.matches:
        return {**{outcome: 0.0 for outcome in OUTCOME_CLASSES}, "macro": 0.0}

    per_outcome: dict[str, PrecisionRecallResult] = {}
    for outcome in OUTCOME_CLASSES:
        tp = sum(
            1
            for m in match_result.matches
            if str(m.predicted.outcome) == outcome and str(m.ground_truth.outcome) == outcome
        )
        fp = sum(
            1
            for m in match_result.matches
            if str(m.predicted.outcome) == outcome and str(m.ground_truth.outcome) != outcome
        )
        fn = sum(
            1
            for m in match_result.matches
            if str(m.predicted.outcome) != outcome and str(m.ground_truth.outcome) == outcome
        )
        per_outcome[outcome] = _precision_recall_f1(tp, fp, fn)

    result = {outcome: per_outcome[outcome].f1 for outcome in OUTCOME_CLASSES}
    result["macro"] = round(sum(result.values()) / len(OUTCOME_CLASSES), 4)
    return result


def confidence_calibration(
    predictions: list[AnalysisEventPrediction], match_result: MatchResult, num_bins: int = 10
) -> dict[str, float]:
    """Expected Calibration Error + Brier score over every prediction's
    confidence vs. whether it was a true positive (`match_result`)."""
    matched_predictions = {id(m.predicted) for m in match_result.matches}
    confidences = np.array([p.confidence for p in predictions])
    correctness = np.array([1.0 if id(p) in matched_predictions else 0.0 for p in predictions])
    return {
        "expected_calibration_error": expected_calibration_error(
            confidences, correctness, num_bins
        ),
        "brier_score": brier_score(confidences, correctness),
    }


def confusion_matrix(match_result: MatchResult) -> dict[str, dict[str, int]]:
    """`{predicted_family: {ground_truth_family: count}}`, plus a `"none"`
    row (predictions with no matching ground truth -- false positives) and
    `"none"` column (ground-truth events with no matching prediction --
    false negatives)."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for match in match_result.matches:
        matrix[match.predicted.family.value][match.ground_truth.family.value] += 1
    for fp in match_result.false_positives:
        matrix[fp.family.value]["none"] += 1
    for fn in match_result.false_negatives:
        matrix["none"][fn.family.value] += 1
    return {row: dict(cols) for row, cols in matrix.items()}


@dataclass(slots=True, frozen=True)
class EvaluationReport:
    event_precision_recall: PrecisionRecallResult
    per_class_precision_recall: dict[str, PrecisionRecallResult]
    macro_f1: float
    temporal_map: TemporalMapResult
    boundary_error: BoundaryErrorResult
    outcome_f1: dict[str, float]
    calibration: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "event_precision_recall": asdict(self.event_precision_recall),
            "per_class_precision_recall": {
                k: asdict(v) for k, v in self.per_class_precision_recall.items()
            },
            "macro_f1": self.macro_f1,
            "temporal_map": {
                "ap_by_threshold": self.temporal_map.ap_by_threshold,
                "map_by_threshold": self.temporal_map.map_by_threshold,
                "mean_map": self.temporal_map.mean_map,
            },
            "boundary_error": asdict(self.boundary_error),
            "outcome_f1": self.outcome_f1,
            "calibration": self.calibration,
            "confusion_matrix": self.confusion_matrix,
        }


def evaluate(
    predictions: list[AnalysisEventPrediction],
    ground_truth: list[TrickEventAnnotation],
    tiou_threshold: float = 0.5,
    tiou_thresholds_map: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    calibration_bins: int = 10,
) -> EvaluationReport:
    """Runs every metric in this module and bundles the results into one
    report -- the single entry point `cli.py`'s `evaluate` command and
    `train.py`'s validation loop both use."""
    match_result = match_events(predictions, ground_truth, tiou_threshold)
    per_class = per_class_precision_recall(predictions, ground_truth, tiou_threshold)
    return EvaluationReport(
        event_precision_recall=event_precision_recall(match_result),
        per_class_precision_recall=per_class,
        macro_f1=macro_f1(per_class),
        temporal_map=temporal_map(predictions, ground_truth, tiou_thresholds_map),
        boundary_error=boundary_error(match_result),
        outcome_f1=outcome_classification_f1(match_result),
        calibration=confidence_calibration(predictions, match_result, calibration_bins),
        confusion_matrix=confusion_matrix(match_result),
    )


def evaluate_detector(
    detector: Any,
    samples: Sequence[TrainingSample],
    tiou_threshold: float = 0.5,
    tiou_thresholds_map: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    calibration_bins: int = 10,
) -> EvaluationReport:
    """Runs `detector.predict(sample.features)` (the
    `interfaces.TemporalEventDetector` shape -- `detector` itself is
    untyped here rather than importing that Protocol, since both the
    always-available `baselines.py` detectors and ad hoc in-training model
    wrappers satisfy it structurally without needing to import from
    `interfaces`) over every one of `samples`, offsetting each clip's
    predictions and ground truth by a clip-specific millisecond offset
    before pooling into one `evaluate()` call -- exactly the "multi-clip
    aggregator" this module's docstring describes, generalized to any
    detector rather than one specific model. The single code path
    `train.py`'s post-training reporting and `cli.py`'s `evaluate`/
    `compare-baselines` commands both use."""
    all_predictions: list[AnalysisEventPrediction] = []
    all_ground_truth: list[TrickEventAnnotation] = []
    for index, sample in enumerate(samples):
        offset_ms = index * _CLIP_OFFSET_MS
        predictions, _deductions = detector.predict(sample.features)
        all_predictions.extend(
            replace(
                prediction,
                start_ms=prediction.start_ms + offset_ms,
                end_ms=prediction.end_ms + offset_ms,
            )
            for prediction in predictions
        )
        all_ground_truth.extend(
            event.model_copy(
                update={"start_ms": event.start_ms + offset_ms, "end_ms": event.end_ms + offset_ms}
            )
            for event in sample.trick_events
        )
    return evaluate(
        all_predictions,
        all_ground_truth,
        tiou_threshold=tiou_threshold,
        tiou_thresholds_map=tiou_thresholds_map,
        calibration_bins=calibration_bins,
    )
