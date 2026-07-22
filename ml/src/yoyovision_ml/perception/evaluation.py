"""Perception evaluation metrics (Prompt B): detector precision/recall,
centre-point pixel error, normalized centre error, track coverage, longest
missing interval, and interpolation rate.

Ground truth is represented as a small, dependency-light `GroundTruthFrame`
list here rather than importing `dataset.schema.YoyoFrameAnnotation`
directly into every function signature -- `ground_truth_from_dataset_track`
is the one conversion point from Prompt A's annotation schema, so these
metric functions stay usable against any ground-truth source (e.g.
hand-built fixtures in tests) without a hard dependency on the dataset
package's Pydantic models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from yoyovision_ml.domain import BoundingBox, Detection, Track


@dataclass(slots=True, frozen=True)
class GroundTruthFrame:
    frame_ms: int
    point: tuple[float, float] | None = None
    bbox: BoundingBox | None = None
    #: Whether a real yo-yo position is *expected* to be observable/labelled
    #: at this frame (i.e. not fully_occluded/outside_frame/unlabelled) --
    #: only these frames count toward recall's denominator.
    expected_visible: bool = True


@runtime_checkable
class _NormalizedPointLike(Protocol):
    x: float
    y: float


@runtime_checkable
class _NormalizedBBoxLike(Protocol):
    x: float
    y: float
    width: float
    height: float


@runtime_checkable
class _YoyoFrameAnnotationLike(Protocol):
    """Structural shape of `dataset.schema.YoyoFrameAnnotation` (Prompt A)
    this module actually reads -- expressed as a `Protocol` (duck-typed)
    rather than importing the dataset schema's Pydantic type directly, so
    this module has no hard import-time dependency on the `dataset` package.
    """

    frame_ms: int
    point: _NormalizedPointLike | None
    bbox: _NormalizedBBoxLike | None
    visibility: object


def ground_truth_from_dataset_track(
    frames: list[_YoyoFrameAnnotationLike],
) -> list[GroundTruthFrame]:
    """Converts `dataset.schema.YoyoFrameAnnotation` rows (Prompt A) into `GroundTruthFrame`s."""
    result: list[GroundTruthFrame] = []
    for row in frames:
        point: tuple[float, float] | None = None
        bbox: BoundingBox | None = None
        if row.point is not None:
            point = (float(row.point.x), float(row.point.y))
        if row.bbox is not None:
            bbox = BoundingBox(
                x=float(row.bbox.x),
                y=float(row.bbox.y),
                width=float(row.bbox.width),
                height=float(row.bbox.height),
            )
            if point is None:
                point = (bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0)
        expected_visible = str(row.visibility) not in {
            "fully_occluded",
            "outside_frame",
            "unlabelled",
        }
        result.append(
            GroundTruthFrame(
                frame_ms=int(row.frame_ms),
                point=point,
                bbox=bbox,
                expected_visible=expected_visible,
            )
        )
    return result


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = b.x + b.width, b.y + b.height
    inter_x1, inter_y1 = max(a.x, b.x), max(a.y, b.y)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 1e-12 else 0.0


def _point_in_bbox(point: tuple[float, float], bbox: BoundingBox) -> bool:
    return bbox.x <= point[0] <= bbox.x + bbox.width and bbox.y <= point[1] <= bbox.y + bbox.height


@dataclass(slots=True, frozen=True)
class PrecisionRecallResult:
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int


def detector_precision_recall(
    predictions: list[Detection], ground_truth: list[GroundTruthFrame], iou_threshold: float = 0.5
) -> PrecisionRecallResult:
    """Frame-level precision/recall for a single-object (yo-yo) detector.

    A ground-truth frame with `expected_visible=True` is a true positive if
    any prediction at that `frame_ms` matches (bbox IoU >= `iou_threshold`,
    or the prediction's bbox contains the GT point when GT has no bbox);
    otherwise it is a false negative. Any prediction at a frame with no
    matching, expected-visible ground truth is a false positive.
    """
    predictions_by_ms: dict[int, list[Detection]] = {}
    for prediction in predictions:
        predictions_by_ms.setdefault(prediction.frame_ms, []).append(prediction)

    true_positives = false_negatives = 0
    matched_ms: set[int] = set()

    for gt in ground_truth:
        if not gt.expected_visible:
            continue
        candidates = predictions_by_ms.get(gt.frame_ms, [])
        matched = False
        for prediction in candidates:
            bbox_matches = (
                gt.bbox is not None and bbox_iou(prediction.bbox, gt.bbox) >= iou_threshold
            )
            point_matches = gt.point is not None and _point_in_bbox(gt.point, prediction.bbox)
            if bbox_matches or point_matches:
                matched = True
            if matched:
                break
        if matched:
            true_positives += 1
            matched_ms.add(gt.frame_ms)
        else:
            false_negatives += 1

    expected_ms = {g.frame_ms for g in ground_truth if g.expected_visible}
    false_positives = sum(
        1 for ms in predictions_by_ms if ms in expected_ms and ms not in matched_ms
    )
    # Predictions at frames with no ground-truth entry at all are also
    # unsupported detections; count those too.
    gt_ms = {g.frame_ms for g in ground_truth}
    false_positives += sum(1 for ms in predictions_by_ms if ms not in gt_ms)

    denom_p = true_positives + false_positives
    denom_r = true_positives + false_negatives
    return PrecisionRecallResult(
        precision=round(true_positives / denom_p, 4) if denom_p else 0.0,
        recall=round(true_positives / denom_r, 4) if denom_r else 0.0,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


@dataclass(slots=True, frozen=True)
class PixelErrorResult:
    mean: float
    median: float
    p95: float
    matched_frames: int


def _error_stats(errors: list[float]) -> PixelErrorResult:
    if not errors:
        return PixelErrorResult(mean=0.0, median=0.0, p95=0.0, matched_frames=0)
    sorted_errors = sorted(errors)
    n = len(sorted_errors)
    median = (
        sorted_errors[n // 2] if n % 2 else (sorted_errors[n // 2 - 1] + sorted_errors[n // 2]) / 2
    )
    p95_idx = min(n - 1, math.ceil(0.95 * n) - 1)
    return PixelErrorResult(
        mean=round(sum(sorted_errors) / n, 4),
        median=round(median, 4),
        p95=round(sorted_errors[p95_idx], 4),
        matched_frames=n,
    )


def _track_centers(tracks: list[Track]) -> dict[int, tuple[float, float]]:
    return {
        t.frame_ms: (t.bbox.x + t.bbox.width / 2.0, t.bbox.y + t.bbox.height / 2.0) for t in tracks
    }


def normalized_centre_error(
    tracks: list[Track], ground_truth: list[GroundTruthFrame]
) -> PixelErrorResult:
    """Centre-point error in normalized `[0, 1]` image-plane units."""
    predicted = _track_centers(tracks)
    errors = [
        math.hypot(predicted[gt.frame_ms][0] - gt.point[0], predicted[gt.frame_ms][1] - gt.point[1])
        for gt in ground_truth
        if gt.point is not None and gt.frame_ms in predicted
    ]
    return _error_stats(errors)


def centre_point_pixel_error(
    tracks: list[Track], ground_truth: list[GroundTruthFrame], width: int, height: int
) -> PixelErrorResult:
    """Centre-point error in pixels, given the video's `width`/`height`."""
    normalized = normalized_centre_error(tracks, ground_truth)
    scale = math.hypot(width, height)
    return PixelErrorResult(
        mean=round(normalized.mean * scale, 2),
        median=round(normalized.median * scale, 2),
        p95=round(normalized.p95 * scale, 2),
        matched_frames=normalized.matched_frames,
    )


def track_coverage(tracks: list[Track], expected_frame_ms: list[int]) -> float:
    """Fraction of `expected_frame_ms` timestamps that have a (possibly
    interpolated) track output -- 1.0 means every expected frame is covered.
    """
    if not expected_frame_ms:
        return 0.0
    covered_ms = {t.frame_ms for t in tracks}
    return round(sum(1 for ms in expected_frame_ms if ms in covered_ms) / len(expected_frame_ms), 4)


def longest_missing_interval(tracks: list[Track], expected_frame_ms: list[int]) -> int:
    """Longest continuous gap (in ms) among `expected_frame_ms` with no track output."""
    if not expected_frame_ms:
        return 0
    covered_ms = {t.frame_ms for t in tracks}
    sorted_expected = sorted(expected_frame_ms)

    longest = 0
    gap_start: int | None = None
    for ms in sorted_expected:
        if ms in covered_ms:
            if gap_start is not None:
                longest = max(longest, ms - gap_start)
                gap_start = None
        elif gap_start is None:
            gap_start = ms
    if gap_start is not None:
        longest = max(longest, sorted_expected[-1] - gap_start)
    return longest


def interpolation_rate(tracks: list[Track]) -> float:
    """Fraction of emitted track frames that were gap-filled (not from a
    real detection) -- see `Track.interpolated`."""
    if not tracks:
        return 0.0
    return round(sum(1 for t in tracks if t.interpolated) / len(tracks), 4)
