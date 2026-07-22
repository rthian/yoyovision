from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from yoyovision_ml.dataset.schema import AnnotationProvenance, TrickEventAnnotation
from yoyovision_ml.domain import EventFamily, FeatureFrame, FeatureSet, Outcome, Source
from yoyovision_ml.events.labels import CLASS_TO_INDEX, NUM_CLASSES
from yoyovision_ml.events.windowing import (
    NormalizationStats,
    build_frame_targets,
    feature_matrix,
    fit_normalization,
    frame_timestamps_ms,
    slice_windows,
)

_FEATURE_NAMES = ("feature_a", "feature_b")


def _feature_set(rows: list[tuple[int, dict[str, float]]], fps: float = 30.0) -> FeatureSet:
    frames = tuple(FeatureFrame(frame_ms=ms, values=values) for ms, values in rows)
    return FeatureSet(frames=frames, feature_names=_FEATURE_NAMES, fps=fps)


def _event(
    family: EventFamily,
    start_ms: int,
    end_ms: int,
    outcome: Outcome = Outcome.SUCCESS,
    event_id: str = "event-0",
) -> TrickEventAnnotation:
    return TrickEventAnnotation(
        event_id=event_id,
        label=family.value,
        family=family,
        start_ms=start_ms,
        end_ms=end_ms,
        outcome=outcome,
        provenance=AnnotationProvenance(
            annotator_id="test",
            source=Source.HUMAN,
            annotated_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
    )


# --------------------------------------------------------------------------- #
# feature_matrix / frame_timestamps_ms
# --------------------------------------------------------------------------- #
def test_feature_matrix_reads_named_columns_in_order() -> None:
    features = _feature_set(
        [(0, {"feature_a": 1.0, "feature_b": 2.0}), (100, {"feature_a": 3.0, "feature_b": 4.0})]
    )
    matrix = feature_matrix(features, _FEATURE_NAMES)
    assert matrix.shape == (2, 2)
    assert np.array_equal(matrix, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_feature_matrix_fills_missing_column_with_zero() -> None:
    features = _feature_set([(0, {"feature_a": 5.0})])
    matrix = feature_matrix(features, _FEATURE_NAMES)
    assert matrix[0, 0] == 5.0
    assert matrix[0, 1] == 0.0


def test_feature_matrix_fills_nan_value_with_zero() -> None:
    features = _feature_set([(0, {"feature_a": float("nan"), "feature_b": 1.0})])
    matrix = feature_matrix(features, _FEATURE_NAMES)
    assert matrix[0, 0] == 0.0
    assert matrix[0, 1] == 1.0


def test_frame_timestamps_ms_returns_int64_array_of_frame_ms() -> None:
    features = _feature_set([(0, {}), (33, {}), (66, {})])
    timestamps = frame_timestamps_ms(features)
    assert timestamps.dtype == np.int64
    assert timestamps.tolist() == [0, 33, 66]


# --------------------------------------------------------------------------- #
# NormalizationStats / fit_normalization
# --------------------------------------------------------------------------- #
def test_normalization_stats_apply_computes_zscore() -> None:
    stats = NormalizationStats(mean=np.array([1.0, 2.0]), std=np.array([2.0, 4.0]))
    matrix = np.array([[3.0, 6.0], [1.0, 2.0]])
    normalized = stats.apply(matrix)
    assert np.allclose(normalized, np.array([[1.0, 1.0], [0.0, 0.0]]))


def test_normalization_stats_to_dict_and_from_dict_round_trip() -> None:
    stats = NormalizationStats(mean=np.array([1.0, 2.0]), std=np.array([3.0, 4.0]))
    restored = NormalizationStats.from_dict(stats.to_dict())
    assert np.allclose(restored.mean, stats.mean)
    assert np.allclose(restored.std, stats.std)


def test_fit_normalization_computes_mean_and_std_across_all_matrices() -> None:
    matrices = [np.array([[0.0], [2.0]]), np.array([[4.0]])]
    stats = fit_normalization(matrices)
    assert stats.mean == pytest.approx(2.0)
    assert stats.std == pytest.approx(np.std([0.0, 2.0, 4.0]))


def test_fit_normalization_floors_std_for_constant_column() -> None:
    matrices = [np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])]
    stats = fit_normalization(matrices)
    assert stats.std[0] == 1.0  # floored, not 0.0 -- would otherwise divide by ~0
    assert stats.std[1] > 0.0


def test_fit_normalization_raises_on_empty_matrix_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fit_normalization([])


# --------------------------------------------------------------------------- #
# build_frame_targets
# --------------------------------------------------------------------------- #
def test_build_frame_targets_marks_multi_hot_class_span() -> None:
    frame_ms = np.array([0, 100, 200, 300, 400])
    event = _event(EventFamily.HOP, start_ms=100, end_ms=300)
    class_targets, _start, _end, _outcome = build_frame_targets(frame_ms, (event,))
    hop_idx = CLASS_TO_INDEX[EventFamily.HOP]
    assert class_targets[:, hop_idx].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]


def test_build_frame_targets_allows_overlapping_labels_from_different_classes() -> None:
    """Prompt C: "Allow overlapping labels where valid.\""""
    frame_ms = np.array([0, 100, 200, 300, 400])
    hop = _event(EventFamily.HOP, start_ms=0, end_ms=300, event_id="hop-0")
    slack = _event(EventFamily.SLACK, start_ms=100, end_ms=400, event_id="slack-0")
    class_targets, _start, _end, _outcome = build_frame_targets(frame_ms, (hop, slack))

    hop_idx = CLASS_TO_INDEX[EventFamily.HOP]
    slack_idx = CLASS_TO_INDEX[EventFamily.SLACK]
    # Frames 100, 200, 300 fall inside *both* spans -- both class columns must
    # be active simultaneously (multi-hot, not one-hot).
    assert class_targets[1, hop_idx] == 1.0
    assert class_targets[1, slack_idx] == 1.0
    assert class_targets[3, hop_idx] == 1.0
    assert class_targets[3, slack_idx] == 1.0


def test_build_frame_targets_sets_outcome_index_within_event_span() -> None:
    frame_ms = np.array([0, 100, 200, 300])
    event = _event(EventFamily.HOP, start_ms=100, end_ms=200, outcome=Outcome.MISS)
    _class, _start, _end, outcome_targets = build_frame_targets(frame_ms, (event,))
    assert outcome_targets.tolist() == [-1, 1, 1, -1]  # OUTCOME_CLASSES.index("miss") == 1


def test_build_frame_targets_leaves_outcome_negative_one_outside_any_event() -> None:
    frame_ms = np.array([0, 100, 200])
    _class, _start, _end, outcome_targets = build_frame_targets(frame_ms, ())
    assert outcome_targets.tolist() == [-1, -1, -1]


def test_build_frame_targets_spikes_start_and_end_at_nearest_frames() -> None:
    frame_ms = np.array([0, 90, 105, 150, 195, 210, 300])
    event = _event(EventFamily.HOP, start_ms=100, end_ms=200)
    _class, start_targets, end_targets, _outcome = build_frame_targets(frame_ms, (event,))
    hop_idx = CLASS_TO_INDEX[EventFamily.HOP]
    # Frames inside [100, 200] are indices 2..4 (105, 150, 195); start_ms=100 is
    # nearest to frame_ms[2]=105, end_ms=200 is nearest to frame_ms[4]=195.
    assert start_targets[2, hop_idx] == 1.0
    assert end_targets[4, hop_idx] == 1.0
    assert start_targets[:, hop_idx].sum() == 1.0
    assert end_targets[:, hop_idx].sum() == 1.0


def test_build_frame_targets_skips_events_outside_the_twenty_prompt_c_classes() -> None:
    """Equipment families (e.g. yoyo_stop) are not in `CLASS_TO_INDEX`; such an
    event must be silently skipped rather than raising or corrupting targets."""
    frame_ms = np.array([0, 100, 200])
    event = _event(EventFamily.YOYO_STOP, start_ms=0, end_ms=200)
    class_targets, _start, _end, outcome_targets = build_frame_targets(frame_ms, (event,))
    assert class_targets.shape == (3, NUM_CLASSES)
    assert not class_targets.any()
    assert outcome_targets.tolist() == [-1, -1, -1]


def test_build_frame_targets_skips_event_with_no_frame_inside_its_span() -> None:
    frame_ms = np.array([0, 1000])
    event = _event(EventFamily.HOP, start_ms=100, end_ms=200)
    class_targets, start_targets, end_targets, outcome_targets = build_frame_targets(
        frame_ms, (event,)
    )
    assert not class_targets.any()
    assert not start_targets.any()
    assert not end_targets.any()
    assert outcome_targets.tolist() == [-1, -1]


# --------------------------------------------------------------------------- #
# slice_windows
# --------------------------------------------------------------------------- #
def _zeros(num_frames: int, num_cols: int = NUM_CLASSES) -> np.ndarray:
    return np.zeros((num_frames, num_cols), dtype=np.float32)


def test_slice_windows_returns_empty_list_for_no_frames() -> None:
    windows = slice_windows(
        np.zeros((0, 2)), np.array([], dtype=np.int64), _zeros(0), _zeros(0), _zeros(0),
        np.array([], dtype=np.int64), window_ms=1000, stride_ms=500,
    )
    assert windows == []


def test_slice_windows_returns_single_window_when_clip_shorter_than_window_ms() -> None:
    frame_ms = np.array([0, 100, 200])
    matrix = np.arange(6, dtype=np.float64).reshape(3, 2)
    windows = slice_windows(
        matrix, frame_ms, _zeros(3), _zeros(3), _zeros(3), np.zeros(3, dtype=np.int64),
        window_ms=4000, stride_ms=2000,
    )
    assert len(windows) == 1
    assert windows[0].frame_ms.tolist() == [0, 100, 200]
    assert np.array_equal(windows[0].features, matrix)


def test_slice_windows_produces_multiple_overlapping_windows_for_a_long_clip() -> None:
    frame_ms = np.arange(0, 10000, 100)  # 100 frames, 0..9900ms
    num_frames = len(frame_ms)
    matrix = np.zeros((num_frames, 1))
    windows = slice_windows(
        matrix, frame_ms, _zeros(num_frames), _zeros(num_frames), _zeros(num_frames),
        np.zeros(num_frames, dtype=np.int64), window_ms=4000, stride_ms=2000,
    )
    assert len(windows) > 1
    for window in windows:
        span_ms = int(window.frame_ms[-1] - window.frame_ms[0])
        assert span_ms <= 4000
    # Every window must actually contain more than one frame.
    assert all(len(window.frame_ms) > 1 for window in windows)


def test_slice_windows_last_window_reaches_the_end_of_the_clip() -> None:
    frame_ms = np.arange(0, 10000, 100)
    num_frames = len(frame_ms)
    matrix = np.zeros((num_frames, 1))
    windows = slice_windows(
        matrix, frame_ms, _zeros(num_frames), _zeros(num_frames), _zeros(num_frames),
        np.zeros(num_frames, dtype=np.int64), window_ms=4000, stride_ms=2000,
    )
    assert windows[-1].frame_ms[-1] == frame_ms[-1]
