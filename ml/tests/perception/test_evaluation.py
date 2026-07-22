from __future__ import annotations

from dataclasses import dataclass

import pytest

from yoyovision_ml.domain import BoundingBox, Detection, Track
from yoyovision_ml.perception.evaluation import (
    GroundTruthFrame,
    bbox_iou,
    centre_point_pixel_error,
    detector_precision_recall,
    ground_truth_from_dataset_track,
    interpolation_rate,
    longest_missing_interval,
    normalized_centre_error,
    track_coverage,
)


def _bbox(cx: float, cy: float, size: float = 0.1) -> BoundingBox:
    return BoundingBox(x=cx - size / 2.0, y=cy - size / 2.0, width=size, height=size)


def _detection(frame_ms: int, cx: float, cy: float, size: float = 0.1) -> Detection:
    return Detection(
        frame_ms=frame_ms,
        bbox=_bbox(cx, cy, size),
        confidence=0.9,
        class_label="yoyo",
        model_name="test",
        model_version="0",
    )


def _track(frame_ms: int, cx: float, cy: float, interpolated: bool = False) -> Track:
    return Track(
        track_id="track-0",
        frame_ms=frame_ms,
        bbox=_bbox(cx, cy),
        confidence=0.8,
        class_label="yoyo",
        interpolated=interpolated,
    )


# --------------------------------------------------------------------------- #
# bbox_iou
# --------------------------------------------------------------------------- #
def test_bbox_iou_identical_boxes_is_one() -> None:
    box = _bbox(0.5, 0.5)
    assert bbox_iou(box, box) == pytest.approx(1.0)


def test_bbox_iou_disjoint_boxes_is_zero() -> None:
    a = BoundingBox(x=0.0, y=0.0, width=0.1, height=0.1)
    b = BoundingBox(x=0.9, y=0.9, width=0.1, height=0.1)
    assert bbox_iou(a, b) == 0.0


def test_bbox_iou_partial_overlap_is_between_zero_and_one() -> None:
    a = BoundingBox(x=0.0, y=0.0, width=0.2, height=0.2)
    b = BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2)
    iou = bbox_iou(a, b)
    assert 0.0 < iou < 1.0


# --------------------------------------------------------------------------- #
# detector_precision_recall
# --------------------------------------------------------------------------- #
def test_precision_recall_perfect_match_is_one_one() -> None:
    predictions = [_detection(0, 0.5, 0.5), _detection(100, 0.5, 0.5)]
    ground_truth = [
        GroundTruthFrame(frame_ms=0, bbox=_bbox(0.5, 0.5)),
        GroundTruthFrame(frame_ms=100, bbox=_bbox(0.5, 0.5)),
    ]
    result = detector_precision_recall(predictions, ground_truth)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_precision_recall_missed_detection_lowers_recall() -> None:
    predictions = [_detection(0, 0.5, 0.5)]
    ground_truth = [
        GroundTruthFrame(frame_ms=0, bbox=_bbox(0.5, 0.5)),
        GroundTruthFrame(frame_ms=100, bbox=_bbox(0.5, 0.5)),
    ]
    result = detector_precision_recall(predictions, ground_truth)
    assert result.recall == 0.5
    assert result.false_negatives == 1


def test_precision_recall_spurious_prediction_lowers_precision() -> None:
    predictions = [_detection(0, 0.5, 0.5), _detection(999, 0.1, 0.1)]
    ground_truth = [GroundTruthFrame(frame_ms=0, bbox=_bbox(0.5, 0.5))]
    result = detector_precision_recall(predictions, ground_truth)
    assert result.precision == 0.5
    assert result.false_positives == 1


def test_precision_recall_uses_point_in_bbox_fallback_when_gt_has_no_bbox() -> None:
    predictions = [_detection(0, 0.5, 0.5, size=0.2)]
    ground_truth = [GroundTruthFrame(frame_ms=0, point=(0.52, 0.52))]
    result = detector_precision_recall(predictions, ground_truth)
    assert result.true_positives == 1
    assert result.recall == 1.0


def test_precision_recall_ignores_frames_not_expected_visible() -> None:
    predictions: list[Detection] = []
    ground_truth = [GroundTruthFrame(frame_ms=0, bbox=_bbox(0.5, 0.5), expected_visible=False)]
    result = detector_precision_recall(predictions, ground_truth)
    # Nothing expected -> no true/false positives or negatives to count.
    assert result.true_positives == 0
    assert result.false_negatives == 0
    assert result.precision == 0.0
    assert result.recall == 0.0


def test_precision_recall_empty_inputs_returns_zero_scores() -> None:
    result = detector_precision_recall([], [])
    assert result.precision == 0.0
    assert result.recall == 0.0


# --------------------------------------------------------------------------- #
# centre error metrics
# --------------------------------------------------------------------------- #
def test_normalized_centre_error_zero_when_tracks_match_ground_truth() -> None:
    tracks = [_track(0, 0.5, 0.5)]
    ground_truth = [GroundTruthFrame(frame_ms=0, point=(0.5, 0.5))]
    result = normalized_centre_error(tracks, ground_truth)
    assert result.mean == 0.0
    assert result.matched_frames == 1


def test_normalized_centre_error_reports_nonzero_for_offset_tracks() -> None:
    tracks = [_track(0, 0.6, 0.5)]
    ground_truth = [GroundTruthFrame(frame_ms=0, point=(0.5, 0.5))]
    result = normalized_centre_error(tracks, ground_truth)
    assert result.mean > 0.0


def test_normalized_centre_error_skips_unmatched_frames() -> None:
    tracks = [_track(0, 0.5, 0.5)]
    ground_truth = [GroundTruthFrame(frame_ms=999, point=(0.5, 0.5))]
    result = normalized_centre_error(tracks, ground_truth)
    assert result.matched_frames == 0
    assert result.mean == 0.0


def test_centre_point_pixel_error_scales_by_frame_diagonal() -> None:
    tracks = [_track(0, 0.6, 0.5)]
    ground_truth = [GroundTruthFrame(frame_ms=0, point=(0.5, 0.5))]
    normalized = normalized_centre_error(tracks, ground_truth)
    pixel = centre_point_pixel_error(tracks, ground_truth, width=1000, height=1000)
    import math

    assert pixel.mean == round(normalized.mean * math.hypot(1000, 1000), 2)


def test_error_stats_median_and_p95_for_multiple_frames() -> None:
    tracks = [_track(0, 0.5, 0.5), _track(100, 0.6, 0.5), _track(200, 0.9, 0.5)]
    ground_truth = [
        GroundTruthFrame(frame_ms=0, point=(0.5, 0.5)),
        GroundTruthFrame(frame_ms=100, point=(0.5, 0.5)),
        GroundTruthFrame(frame_ms=200, point=(0.5, 0.5)),
    ]
    result = normalized_centre_error(tracks, ground_truth)
    assert result.matched_frames == 3
    assert result.mean > 0.0
    assert result.p95 >= result.median


# --------------------------------------------------------------------------- #
# coverage / gaps / interpolation
# --------------------------------------------------------------------------- #
def test_track_coverage_full_when_every_expected_frame_has_a_track() -> None:
    tracks = [_track(0, 0.5, 0.5), _track(100, 0.5, 0.5)]
    assert track_coverage(tracks, [0, 100]) == 1.0


def test_track_coverage_partial() -> None:
    tracks = [_track(0, 0.5, 0.5)]
    assert track_coverage(tracks, [0, 100, 200]) == round(1 / 3, 4)


def test_track_coverage_empty_expected_is_zero() -> None:
    assert track_coverage([_track(0, 0.5, 0.5)], []) == 0.0


def test_longest_missing_interval_finds_largest_gap() -> None:
    tracks = [_track(0, 0.5, 0.5), _track(400, 0.5, 0.5)]
    expected = [0, 100, 200, 300, 400]
    # Missing 100/200/300 -> measured from first missing (100) to next covered
    # (400), matching `longest_missing_interval`'s "distance to next covered
    # frame" definition.
    assert longest_missing_interval(tracks, expected) == 300


def test_longest_missing_interval_zero_when_fully_covered() -> None:
    tracks = [_track(ms, 0.5, 0.5) for ms in (0, 100, 200)]
    assert longest_missing_interval(tracks, [0, 100, 200]) == 0


def test_longest_missing_interval_trailing_gap() -> None:
    tracks = [_track(0, 0.5, 0.5)]
    expected = [0, 100, 200, 300]
    # Trailing gap is measured from the first missing frame (100) to the
    # last expected frame (300).
    assert longest_missing_interval(tracks, expected) == 200


def test_interpolation_rate_counts_interpolated_fraction() -> None:
    tracks = [_track(0, 0.5, 0.5, interpolated=False), _track(100, 0.5, 0.5, interpolated=True)]
    assert interpolation_rate(tracks) == 0.5


def test_interpolation_rate_empty_tracks_is_zero() -> None:
    assert interpolation_rate([]) == 0.0


# --------------------------------------------------------------------------- #
# ground_truth_from_dataset_track
# --------------------------------------------------------------------------- #
@dataclass
class _FakeNormalizedPoint:
    x: float
    y: float


@dataclass
class _FakeNormalizedBBox:
    x: float
    y: float
    width: float
    height: float


@dataclass
class _FakeYoyoFrameAnnotation:
    frame_ms: int
    point: _FakeNormalizedPoint | None
    bbox: _FakeNormalizedBBox | None
    visibility: str


def test_ground_truth_from_dataset_track_converts_point_and_bbox() -> None:
    rows = [
        _FakeYoyoFrameAnnotation(
            frame_ms=0, point=_FakeNormalizedPoint(0.5, 0.5), bbox=None, visibility="visible"
        ),
        _FakeYoyoFrameAnnotation(
            frame_ms=100,
            point=None,
            bbox=_FakeNormalizedBBox(0.4, 0.4, 0.2, 0.2),
            visibility="partially_occluded",
        ),
    ]
    ground_truth = ground_truth_from_dataset_track(rows)  # type: ignore[arg-type]

    assert len(ground_truth) == 2
    assert ground_truth[0].point == (0.5, 0.5)
    assert ground_truth[0].expected_visible is True
    assert ground_truth[1].bbox is not None
    # bbox-derived point should be the bbox center.
    assert ground_truth[1].point == (0.5, 0.5)


def test_ground_truth_from_dataset_track_marks_occluded_frames_not_expected_visible() -> None:
    rows = [
        _FakeYoyoFrameAnnotation(frame_ms=0, point=None, bbox=None, visibility="fully_occluded"),
        _FakeYoyoFrameAnnotation(frame_ms=100, point=None, bbox=None, visibility="outside_frame"),
        _FakeYoyoFrameAnnotation(frame_ms=200, point=None, bbox=None, visibility="unlabelled"),
    ]
    ground_truth = ground_truth_from_dataset_track(rows)  # type: ignore[arg-type]
    assert all(not g.expected_visible for g in ground_truth)
