"""Turns a `FeatureSet` + ground-truth `TrickEventAnnotation`s into the plain
`numpy` arrays the model/training loop consumes.

Deliberately pure `numpy` (no `torch` import) so this module -- and every
test of it -- never requires the optional `torch` extra to be installed;
`events/model.py` converts the arrays this module produces into tensors at
the last possible moment.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from yoyovision_ml.dataset.schema import TrickEventAnnotation
from yoyovision_ml.domain import FeatureSet
from yoyovision_ml.events.labels import CLASS_TO_INDEX, NUM_CLASSES, OUTCOME_CLASSES


def feature_matrix(features: FeatureSet, feature_names: tuple[str, ...]) -> np.ndarray:
    """`(T, F)` matrix in `feature_names` order, one row per `features.frames`
    entry (already time-sorted -- callers must not assume this and should
    sort `features.frames` upstream if the source isn't already ordered).

    Any `feature_names` column absent from a given frame's `values` (e.g. a
    `FeatureSet` produced by the older, thinner `feature_extraction.py`
    rather than Prompt B's full kinematic feature set) is filled with `0.0`
    rather than raising -- this module must degrade gracefully on whatever
    `FeatureSet` a `TemporalEventDetector` caller happens to pass in.
    """
    matrix = np.zeros((len(features.frames), len(feature_names)), dtype=np.float64)
    for row, frame in enumerate(features.frames):
        for col, name in enumerate(feature_names):
            value = frame.values.get(name)
            if value is not None and not np.isnan(value):
                matrix[row, col] = value
    return matrix


def frame_timestamps_ms(features: FeatureSet) -> np.ndarray:
    return np.array([frame.frame_ms for frame in features.frames], dtype=np.int64)


@dataclass(slots=True, frozen=True)
class NormalizationStats:
    """Per-column z-score statistics, fit on a training split only (Prompt
    C's "input feature normalization") and re-applied identically at
    inference time. `std` is floored so a constant training-split column
    (e.g. all-zero `yoyo_interpolated` in a clip with no gaps) never divides
    by ~0."""

    mean: np.ndarray
    std: np.ndarray

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        result: np.ndarray = (matrix - self.mean) / self.std
        return result

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> NormalizationStats:
        return cls(mean=np.array(data["mean"]), std=np.array(data["std"]))


def fit_normalization(matrices: list[np.ndarray]) -> NormalizationStats:
    """Fits per-column mean/std across every row of every matrix in
    `matrices` (typically all training-split clips concatenated)."""
    if not matrices:
        raise ValueError("fit_normalization requires at least one feature matrix")
    stacked = np.concatenate(matrices, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return NormalizationStats(mean=mean, std=std)


def _class_index_for_family(annotation: TrickEventAnnotation) -> int | None:
    return CLASS_TO_INDEX.get(annotation.family)


def build_frame_targets(
    frame_ms: np.ndarray, trick_events: tuple[TrickEventAnnotation, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds the four per-frame training targets for one clip's `(T,)`
    `frame_ms` grid:

    - `class_targets`: `(T, NUM_CLASSES)` multi-hot -- 1.0 for every class
      whose event span covers a frame. Multi-hot (not one-hot) because
      Prompt C explicitly allows overlapping labels.
    - `start_targets` / `end_targets`: `(T, NUM_CLASSES)` -- 1.0 on the frame
      nearest an event's `start_ms`/`end_ms` for that event's class, 0.0
      elsewhere. Sparse "boundary spike" targets for the boundary head.
    - `outcome_targets`: `(T,)` integer outcome-class index (see
      `labels.OUTCOME_CLASSES`), or `-1` on frames not covered by any event
      (the training loop must mask these out of the outcome loss).
    """
    num_frames = len(frame_ms)
    class_targets = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    start_targets = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    end_targets = np.zeros((num_frames, NUM_CLASSES), dtype=np.float32)
    outcome_targets = np.full(num_frames, -1, dtype=np.int64)

    sorted_ms = frame_ms.tolist()
    for event in trick_events:
        class_idx = _class_index_for_family(event)
        if class_idx is None:
            continue  # equipment family or otherwise outside the 20 Prompt C classes

        lo = bisect.bisect_left(sorted_ms, event.start_ms)
        hi = bisect.bisect_right(sorted_ms, event.end_ms)
        if lo >= hi:
            continue  # no frame falls inside this event's span at all
        class_targets[lo:hi, class_idx] = 1.0

        outcome_idx = OUTCOME_CLASSES.index(str(event.outcome))
        outcome_targets[lo:hi] = outcome_idx

        start_frame = min(
            range(lo, hi), key=lambda i: abs(sorted_ms[i] - event.start_ms), default=None
        )
        if start_frame is not None:
            start_targets[start_frame, class_idx] = 1.0
        end_frame = max(
            range(lo, hi), key=lambda i: -abs(sorted_ms[i] - event.end_ms), default=None
        )
        if end_frame is not None:
            end_targets[end_frame, class_idx] = 1.0

    return class_targets, start_targets, end_targets, outcome_targets


@dataclass(slots=True, frozen=True)
class Window:
    """One fixed-length, contiguous slice of a clip's feature/target arrays,
    as produced by `slice_windows` for training."""

    features: np.ndarray  # (window_len, F)
    class_targets: np.ndarray  # (window_len, NUM_CLASSES)
    start_targets: np.ndarray  # (window_len, NUM_CLASSES)
    end_targets: np.ndarray  # (window_len, NUM_CLASSES)
    outcome_targets: np.ndarray  # (window_len,)
    frame_ms: np.ndarray  # (window_len,)


def slice_windows(
    features_matrix: np.ndarray,
    frame_ms: np.ndarray,
    class_targets: np.ndarray,
    start_targets: np.ndarray,
    end_targets: np.ndarray,
    outcome_targets: np.ndarray,
    window_ms: int,
    stride_ms: int,
) -> list[Window]:
    """Slices one clip's frame-indexed arrays into fixed-length `window_ms`
    windows, stepping by `stride_ms` (Prompt C: "configurable temporal
    window"). A clip shorter than `window_ms` yields exactly one window
    covering everything available (never dropped, since short synthetic/real
    clips must still be trainable)."""
    if len(frame_ms) == 0:
        return []

    total_ms = int(frame_ms[-1] - frame_ms[0])
    if total_ms < window_ms:
        return [
            Window(
                features=features_matrix,
                class_targets=class_targets,
                start_targets=start_targets,
                end_targets=end_targets,
                outcome_targets=outcome_targets,
                frame_ms=frame_ms,
            )
        ]

    windows: list[Window] = []
    window_start_ms = int(frame_ms[0])
    clip_end_ms = int(frame_ms[-1])
    while window_start_ms <= clip_end_ms:
        window_end_ms = window_start_ms + window_ms
        lo = bisect.bisect_left(frame_ms.tolist(), window_start_ms)
        hi = bisect.bisect_left(frame_ms.tolist(), window_end_ms)
        if hi - lo > 1:
            windows.append(
                Window(
                    features=features_matrix[lo:hi],
                    class_targets=class_targets[lo:hi],
                    start_targets=start_targets[lo:hi],
                    end_targets=end_targets[lo:hi],
                    outcome_targets=outcome_targets[lo:hi],
                    frame_ms=frame_ms[lo:hi],
                )
            )
        if window_end_ms >= clip_end_ms:
            break
        window_start_ms += stride_ms
    return windows
