from __future__ import annotations

import numpy as np

from yoyovision_ml.domain import EventFamily, Outcome
from yoyovision_ml.events.config import InferenceConfig
from yoyovision_ml.events.decode import decode_predictions
from yoyovision_ml.events.labels import CLASS_TO_INDEX, NUM_CLASSES, NUM_OUTCOMES, OUTCOME_CLASSES


def _uniform_outcome_probs(num_frames: int, outcome: str = "success") -> np.ndarray:
    probs = np.zeros((num_frames, NUM_OUTCOMES))
    probs[:, OUTCOME_CLASSES.index(outcome)] = 1.0
    return probs


def test_decode_predictions_returns_empty_for_no_frames() -> None:
    config = InferenceConfig()
    detections = decode_predictions(
        np.array([]), np.zeros((0, NUM_CLASSES)), np.zeros((0, NUM_OUTCOMES)), "v1", config
    )
    assert detections == []


def test_decode_predictions_detects_a_single_active_class_span() -> None:
    frame_ms = np.array([0, 100, 200, 300, 400])
    class_probs = np.zeros((5, NUM_CLASSES))
    hop_idx = CLASS_TO_INDEX[EventFamily.HOP]
    class_probs[1:4, hop_idx] = 0.9
    config = InferenceConfig(min_event_ms=0, uncertainty_threshold=0.0)

    detections = decode_predictions(
        frame_ms, class_probs, _uniform_outcome_probs(5), "v1", config
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.family == EventFamily.HOP
    assert detection.start_ms == 100
    # Without a boundary head, the end estimate is the frame *after* the last
    # active frame (index `hi`, exclusive-upper-bound convention), i.e. 400ms
    # here, not the last active frame's own timestamp (300ms).
    assert detection.end_ms == 400
    assert detection.confidence == 0.9
    assert detection.outcome == Outcome.SUCCESS
    assert detection.model_version == "v1"


def test_decode_predictions_ignores_frames_below_activation_threshold() -> None:
    frame_ms = np.array([0, 100, 200])
    class_probs = np.full((3, NUM_CLASSES), 0.1)  # below default 0.5 threshold everywhere
    config = InferenceConfig()
    detections = decode_predictions(
        frame_ms, class_probs, _uniform_outcome_probs(3), "v1", config
    )
    assert detections == []


def test_decode_predictions_drops_spans_shorter_than_min_event_ms() -> None:
    frame_ms = np.array([0, 100, 200])
    class_probs = np.zeros((3, NUM_CLASSES))
    hop_idx = CLASS_TO_INDEX[EventFamily.HOP]
    class_probs[1, hop_idx] = 0.9  # only one 100ms-wide frame span
    config = InferenceConfig(min_event_ms=500, uncertainty_threshold=0.0)
    detections = decode_predictions(
        frame_ms, class_probs, _uniform_outcome_probs(3), "v1", config
    )
    assert detections == []


def test_decode_predictions_refines_boundaries_using_start_end_probs() -> None:
    frame_ms = np.array([0, 100, 200, 300, 400, 500])
    class_probs = np.zeros((6, NUM_CLASSES))
    hop_idx = CLASS_TO_INDEX[EventFamily.HOP]
    class_probs[1:5, hop_idx] = 0.9

    start_probs = np.zeros((6, NUM_CLASSES))
    start_probs[2, hop_idx] = 1.0  # sharper start estimate than raw run's first frame (idx 1)
    end_probs = np.zeros((6, NUM_CLASSES))
    end_probs[3, hop_idx] = 1.0  # sharper end estimate than raw run's last frame (idx 4)

    config = InferenceConfig(min_event_ms=0, uncertainty_threshold=0.0)
    detections = decode_predictions(
        frame_ms,
        class_probs,
        _uniform_outcome_probs(6),
        "v1",
        config,
        start_probs=start_probs,
        end_probs=end_probs,
    )
    assert len(detections) == 1
    assert detections[0].start_ms == 200
    assert detections[0].end_ms == 400


def test_decode_predictions_merge_strategy_joins_nearby_same_class_spans() -> None:
    frame_ms = np.array([0, 100, 200, 300, 400, 500, 600])
    class_probs = np.zeros((7, NUM_CLASSES))
    hop_idx = CLASS_TO_INDEX[EventFamily.HOP]
    class_probs[1, hop_idx] = 0.9
    class_probs[4:6, hop_idx] = 0.9  # separate run, gap of 200ms from the first span's end
    config = InferenceConfig(
        min_event_ms=0, uncertainty_threshold=0.0, nms_strategy="merge", merge_gap_ms=250
    )
    detections = decode_predictions(
        frame_ms, class_probs, _uniform_outcome_probs(7), "v1", config
    )
    assert len(detections) == 1
    assert detections[0].start_ms == 100
    # Same exclusive-upper-bound end convention as the single-span case: the
    # second run's raw end is frame_ms[hi]=600, so the merged span ends there.
    assert detections[0].end_ms == 600


def test_decode_predictions_suppress_strategy_keeps_a_single_contiguous_run() -> None:
    frame_ms = np.array([0, 100, 200, 300])
    class_probs = np.zeros((4, NUM_CLASSES))
    hop_idx = CLASS_TO_INDEX[EventFamily.HOP]
    class_probs[0:3, hop_idx] = 0.9
    config = InferenceConfig(
        min_event_ms=0, uncertainty_threshold=0.0, nms_strategy="suppress", nms_iou_threshold=0.3
    )
    detections = decode_predictions(
        frame_ms, class_probs, _uniform_outcome_probs(4), "v1", config
    )
    assert len(detections) == 1


def test_decode_predictions_relabels_low_confidence_as_unknown_technical_element() -> None:
    frame_ms = np.array([0, 100, 200, 300])
    class_probs = np.zeros((4, NUM_CLASSES))
    hop_idx = CLASS_TO_INDEX[EventFamily.HOP]
    class_probs[0:3, hop_idx] = 0.55  # active but below uncertainty_threshold
    config = InferenceConfig(
        min_event_ms=0, uncertainty_threshold=0.9, uncertainty_action="relabel_unknown"
    )
    detections = decode_predictions(
        frame_ms, class_probs, _uniform_outcome_probs(4), "v1", config
    )
    assert len(detections) == 1
    assert detections[0].family == EventFamily.UNKNOWN_TECHNICAL_ELEMENT
    assert detections[0].label == EventFamily.UNKNOWN_TECHNICAL_ELEMENT.value
    assert detections[0].needs_review is False


def test_decode_predictions_flags_low_confidence_for_review_instead_of_relabeling() -> None:
    frame_ms = np.array([0, 100, 200, 300])
    class_probs = np.zeros((4, NUM_CLASSES))
    hop_idx = CLASS_TO_INDEX[EventFamily.HOP]
    class_probs[0:3, hop_idx] = 0.55
    config = InferenceConfig(
        min_event_ms=0, uncertainty_threshold=0.9, uncertainty_action="flag_review"
    )
    detections = decode_predictions(
        frame_ms, class_probs, _uniform_outcome_probs(4), "v1", config
    )
    assert len(detections) == 1
    assert detections[0].family == EventFamily.HOP  # label unchanged
    assert detections[0].needs_review is True


def test_decode_predictions_selects_most_probable_outcome_over_the_event_span() -> None:
    frame_ms = np.array([0, 100, 200])
    class_probs = np.zeros((3, NUM_CLASSES))
    hop_idx = CLASS_TO_INDEX[EventFamily.HOP]
    class_probs[0:2, hop_idx] = 0.9
    outcome_probs = np.zeros((3, NUM_OUTCOMES))
    outcome_probs[0:2, OUTCOME_CLASSES.index("miss")] = 1.0
    config = InferenceConfig(min_event_ms=0, uncertainty_threshold=0.0)
    detections = decode_predictions(frame_ms, class_probs, outcome_probs, "v1", config)
    assert len(detections) == 1
    assert detections[0].outcome == Outcome.MISS


def test_decode_predictions_sorts_detections_by_start_ms() -> None:
    frame_ms = np.array([0, 100, 200, 300, 400, 500])
    class_probs = np.zeros((6, NUM_CLASSES))
    hop_idx = CLASS_TO_INDEX[EventFamily.HOP]
    slack_idx = CLASS_TO_INDEX[EventFamily.SLACK]
    class_probs[3:5, hop_idx] = 0.9  # later span
    class_probs[0:2, slack_idx] = 0.9  # earlier span
    config = InferenceConfig(min_event_ms=0, uncertainty_threshold=0.0)
    detections = decode_predictions(
        frame_ms, class_probs, _uniform_outcome_probs(6), "v1", config
    )
    assert len(detections) == 2
    assert detections[0].start_ms <= detections[1].start_ms
    assert detections[0].family == EventFamily.SLACK
    assert detections[1].family == EventFamily.HOP


def test_decode_predictions_supporting_frame_range_matches_start_and_end() -> None:
    frame_ms = np.array([0, 100, 200, 300])
    class_probs = np.zeros((4, NUM_CLASSES))
    hop_idx = CLASS_TO_INDEX[EventFamily.HOP]
    class_probs[0:3, hop_idx] = 0.9
    config = InferenceConfig(min_event_ms=0, uncertainty_threshold=0.0)
    detections = decode_predictions(
        frame_ms, class_probs, _uniform_outcome_probs(4), "v1", config
    )
    assert detections[0].supporting_frame_range == (detections[0].start_ms, detections[0].end_ms)
