from __future__ import annotations

from yoyovision_ml.domain import BoundingBox, Detection, VisibilityState
from yoyovision_ml.perception.tracker_kalman import KalmanYoyoTracker


def _detection(
    frame_ms: int, center_x: float, center_y: float, confidence: float = 0.9
) -> Detection:
    """`center_x`/`center_y` are the desired bbox *center*, matching how the
    tracker itself reasons about position (`_bbox_center`)."""
    width = height = 0.05
    bbox = BoundingBox(
        x=center_x - width / 2.0, y=center_y - height / 2.0, width=width, height=height
    )
    return Detection(
        frame_ms=frame_ms,
        bbox=bbox,
        confidence=confidence,
        class_label="yoyo",
        model_name="test-detector",
        model_version="0.0.0",
    )


def test_first_detection_initializes_track_at_measured_position() -> None:
    tracker = KalmanYoyoTracker()
    detections = [_detection(0, 0.5, 0.5)]
    tracks = tracker.update(detections, timestamp_ms=0)

    assert len(tracks) == 1
    track = tracks[0]
    assert track.frame_ms == 0
    assert track.visibility == VisibilityState.VISIBLE
    assert track.interpolated is False
    assert track.bbox.x + track.bbox.width / 2.0 == 0.5
    assert track.bbox.y + track.bbox.height / 2.0 == 0.5


def test_tracker_follows_linearly_moving_detections() -> None:
    tracker = KalmanYoyoTracker()
    xs = [0.1, 0.2, 0.3, 0.4, 0.5]
    last_center_x = None
    for i, x in enumerate(xs):
        frame_ms = i * 100
        tracks = tracker.update([_detection(frame_ms, x, 0.5)], timestamp_ms=frame_ms)
        assert len(tracks) == 1
        last_center_x = tracks[0].bbox.x + tracks[0].bbox.width / 2.0

    # After several consistent measurements, the filter should have converged
    # near the true position (allow modest Kalman lag).
    assert last_center_x is not None
    assert abs(last_center_x - 0.5) < 0.05


def test_short_gap_is_bridged_with_interpolated_prediction() -> None:
    tracker = KalmanYoyoTracker(max_gap_ms=500)
    tracker.update([_detection(0, 0.5, 0.5)], timestamp_ms=0)
    tracker.update([_detection(100, 0.55, 0.5)], timestamp_ms=100)

    # No detection at t=200ms, but gap (100ms) is within max_gap_ms.
    tracks = tracker.update([], timestamp_ms=200)

    assert len(tracks) == 1
    track = tracks[0]
    assert track.interpolated is True
    assert track.visibility == VisibilityState.FULLY_OCCLUDED
    assert track.confidence == 0.0


def test_gap_beyond_max_gap_ms_emits_no_track() -> None:
    tracker = KalmanYoyoTracker(max_gap_ms=200)
    tracker.update([_detection(0, 0.5, 0.5)], timestamp_ms=0)

    # Gap of 1000ms far exceeds max_gap_ms=200 -- tracker should give up.
    tracks = tracker.update([], timestamp_ms=1000)

    assert tracks == []


def test_update_out_of_order_raises_value_error() -> None:
    tracker = KalmanYoyoTracker()
    tracker.update([_detection(100, 0.5, 0.5)], timestamp_ms=100)

    try:
        tracker.update([_detection(0, 0.5, 0.5)], timestamp_ms=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-order update")


def test_reset_clears_internal_state() -> None:
    tracker = KalmanYoyoTracker()
    tracker.update([_detection(0, 0.5, 0.5)], timestamp_ms=0)
    tracker.reset()

    # After reset, a fresh detection at frame_ms=0 should behave like the
    # very first update (no out-of-order error, track quality reset to 0).
    tracks = tracker.update([_detection(0, 0.3, 0.3)], timestamp_ms=0)
    assert len(tracks) == 1
    assert tracks[0].bbox.x + tracks[0].bbox.width / 2.0 == 0.3


def test_track_quality_reflects_coverage_and_confidence() -> None:
    tracker = KalmanYoyoTracker(max_gap_ms=500)
    assert tracker.track_quality() == 0.0

    tracker.update([_detection(0, 0.5, 0.5, confidence=1.0)], timestamp_ms=0)
    tracker.update([_detection(100, 0.5, 0.5, confidence=1.0)], timestamp_ms=100)

    quality = tracker.track_quality()
    assert 0.0 < quality <= 1.0

    # A subsequent missed-but-bridged frame (confidence 0.0) should lower quality.
    tracker.update([], timestamp_ms=200)
    lower_quality = tracker.track_quality()
    assert lower_quality < quality


def test_static_camera_uses_lower_default_process_noise() -> None:
    static_tracker = KalmanYoyoTracker(static_camera=True)
    moving_tracker = KalmanYoyoTracker(static_camera=False)
    assert static_tracker.process_noise < moving_tracker.process_noise


def test_explicit_process_noise_overrides_static_camera_default() -> None:
    tracker = KalmanYoyoTracker(static_camera=True, process_noise=0.123)
    assert tracker.process_noise == 0.123
