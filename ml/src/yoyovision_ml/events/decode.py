"""Turns calibrated per-frame probabilities into discrete `EventDetection`s.

Pure `numpy` -- no `torch` dependency -- so both the trained TCN adapter
(`detector_torch.py`) and the always-available baselines (`baselines.py`)
share exactly one decode implementation, and it is fully testable without
the optional `torch` extra installed.
"""

from __future__ import annotations

import numpy as np

from yoyovision_ml.domain import EventFamily, Outcome
from yoyovision_ml.events.config import InferenceConfig
from yoyovision_ml.events.labels import INDEX_TO_CLASS, NUM_CLASSES, OUTCOME_CLASSES
from yoyovision_ml.events.types import EventDetection


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Returns `[(lo, hi_exclusive), ...]` index ranges of contiguous `True`
    runs in a 1-D boolean array."""
    if mask.size == 0 or not mask.any():
        return []
    changes = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    starts = changes[0::2]
    ends = changes[1::2]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _refine_boundary_index(probs_column: np.ndarray, lo: int, hi: int, margin: int) -> int:
    """Picks the frame index (within `[lo - margin, hi + margin)`, clamped to
    the array) with the highest boundary-head probability -- a sharper start/
    end estimate than "first/last frame the classification head fired on"."""
    window_lo = max(0, lo - margin)
    window_hi = min(len(probs_column), hi + margin)
    if window_hi <= window_lo:
        return lo
    window = probs_column[window_lo:window_hi]
    return window_lo + int(np.argmax(window))


def _temporal_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


class _Candidate:
    __slots__ = ("class_idx", "start_ms", "end_ms", "confidence", "outcome_idx")

    def __init__(
        self, class_idx: int, start_ms: int, end_ms: int, confidence: float, outcome_idx: int
    ) -> None:
        self.class_idx = class_idx
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.confidence = confidence
        self.outcome_idx = outcome_idx


def _class_candidates(
    class_idx: int,
    frame_ms: np.ndarray,
    class_probs: np.ndarray,
    outcome_probs: np.ndarray,
    start_probs: np.ndarray | None,
    end_probs: np.ndarray | None,
    config: InferenceConfig,
) -> list[_Candidate]:
    mask = class_probs[:, class_idx] >= config.frame_activation_threshold
    candidates: list[_Candidate] = []
    for lo, hi in _find_runs(mask):
        refined_lo = (
            _refine_boundary_index(start_probs[:, class_idx], lo, hi, margin=3)
            if start_probs is not None
            else lo
        )
        refined_hi_exclusive = (
            _refine_boundary_index(end_probs[:, class_idx], lo, hi, margin=3) + 1
            if end_probs is not None
            else hi
        )
        refined_lo = min(refined_lo, hi - 1)
        refined_hi_exclusive = max(refined_hi_exclusive, refined_lo + 1)

        start_ms = int(frame_ms[refined_lo])
        end_ms = int(frame_ms[min(refined_hi_exclusive, len(frame_ms) - 1)])
        if end_ms - start_ms < config.min_event_ms:
            continue

        confidence = float(class_probs[lo:hi, class_idx].mean())
        mean_outcome = outcome_probs[lo:hi].mean(axis=0)
        outcome_idx = int(np.argmax(mean_outcome))
        candidates.append(_Candidate(class_idx, start_ms, end_ms, confidence, outcome_idx))
    return candidates


def _suppress(candidates: list[_Candidate], iou_threshold: float) -> list[_Candidate]:
    kept: list[_Candidate] = []
    for candidate in sorted(candidates, key=lambda c: -c.confidence):
        if all(
            _temporal_iou((candidate.start_ms, candidate.end_ms), (k.start_ms, k.end_ms))
            < iou_threshold
            for k in kept
        ):
            kept.append(candidate)
    return kept


def _merge(candidates: list[_Candidate], merge_gap_ms: int) -> list[_Candidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c.start_ms)
    merged = [ordered[0]]
    for candidate in ordered[1:]:
        last = merged[-1]
        if candidate.start_ms - last.end_ms <= merge_gap_ms:
            merged[-1] = _Candidate(
                class_idx=last.class_idx,
                start_ms=last.start_ms,
                end_ms=max(last.end_ms, candidate.end_ms),
                confidence=max(last.confidence, candidate.confidence),
                outcome_idx=(
                    candidate.outcome_idx
                    if candidate.confidence > last.confidence
                    else last.outcome_idx
                ),
            )
        else:
            merged.append(candidate)
    return merged


def decode_predictions(
    frame_ms: np.ndarray,
    class_probs: np.ndarray,
    outcome_probs: np.ndarray,
    model_version: str,
    config: InferenceConfig,
    start_probs: np.ndarray | None = None,
    end_probs: np.ndarray | None = None,
) -> list[EventDetection]:
    """Full decode: per-class thresholding -> boundary refinement -> per-class
    NMS/merge (`config.nms_strategy`) -> uncertainty-threshold routing.

    `class_probs`/`outcome_probs`/`start_probs`/`end_probs` must already be
    *calibrated* probabilities (post `calibration.apply_temperature`), all
    shaped `(T, ...)` aligned to `frame_ms`.
    """
    if len(frame_ms) == 0:
        return []

    all_candidates: list[_Candidate] = []
    for class_idx in range(NUM_CLASSES):
        class_candidates = _class_candidates(
            class_idx, frame_ms, class_probs, outcome_probs, start_probs, end_probs, config
        )
        if config.nms_strategy == "merge":
            class_candidates = _merge(class_candidates, config.merge_gap_ms)
        else:
            class_candidates = _suppress(class_candidates, config.nms_iou_threshold)
        all_candidates.extend(class_candidates)

    detections: list[EventDetection] = []
    for candidate in all_candidates:
        family = INDEX_TO_CLASS[candidate.class_idx]
        outcome = Outcome(OUTCOME_CLASSES[candidate.outcome_idx])
        label = family.value
        needs_review = False

        if candidate.confidence < config.uncertainty_threshold:
            if config.uncertainty_action == "relabel_unknown":
                family = EventFamily.UNKNOWN_TECHNICAL_ELEMENT
                label = family.value
            else:
                needs_review = True

        detections.append(
            EventDetection(
                label=label,
                family=family,
                start_ms=candidate.start_ms,
                end_ms=candidate.end_ms,
                outcome=outcome,
                confidence=round(candidate.confidence, 4),
                model_version=model_version,
                supporting_frame_range=(candidate.start_ms, candidate.end_ms),
                needs_review=needs_review,
            )
        )

    detections.sort(key=lambda d: d.start_ms)
    return detections
