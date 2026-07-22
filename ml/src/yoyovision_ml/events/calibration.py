"""Post-hoc confidence calibration via temperature scaling.

Pure `numpy` (no `torch`, no `scipy`) -- deliberately, so calibration is
always available (including for the non-torch `baselines.py` adapters, which
also route their raw scores through `apply_temperature` with `temperature=1.0`
i.e. a no-op) and so `events/train.py` only needs `torch` for the model
itself, not for this one scalar-per-class fit. Temperature scaling (Guo et
al., 2017) rescales logits before the sigmoid: `calibrated = sigmoid(logit /
T)`. `T > 1` softens over-confident predictions; `T < 1` sharpens
under-confident ones. Fit with a 1-D grid search (not gradient descent) since
one scalar per class needs no more than that to be exact enough, and a grid
search has no learning-rate/convergence knobs to get wrong.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-7


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _binary_cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probs, _EPS, 1.0 - _EPS)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, grid: np.ndarray | None = None
) -> float:
    """Grid-searches the scalar temperature `T` minimizing binary
    cross-entropy of `sigmoid(logits / T)` against binary `labels`
    (flattened -- caller decides what population e.g. one class's frames,
    across a validation split, to fit on)."""
    if logits.size == 0:
        return 1.0
    if grid is None:
        grid = np.concatenate([np.linspace(0.05, 1.0, 20), np.linspace(1.0, 10.0, 46)[1:]])

    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in grid:
        probs = sigmoid(logits / temperature)
        loss = _binary_cross_entropy(probs, labels)
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature


def fit_temperature_per_class(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """`logits`/`labels`: `(N, C)` -> `(C,)` independently-fit temperatures,
    one per class column."""
    num_classes = logits.shape[1]
    return np.array([fit_temperature(logits[:, c], labels[:, c]) for c in range(num_classes)])


def apply_temperature(logits: np.ndarray, temperature: np.ndarray | float) -> np.ndarray:
    """Returns calibrated probabilities `sigmoid(logits / temperature)`.
    `temperature` broadcasts against `logits`' last axis when given as a
    per-class `(C,)` array."""
    return sigmoid(logits / temperature)


def expected_calibration_error(
    confidences: np.ndarray, correctness: np.ndarray, num_bins: int = 10
) -> float:
    """Standard ECE: bins detections by confidence into `num_bins` equal-width
    bins, and averages `|mean_confidence - accuracy|` per bin, weighted by
    bin size. `correctness` is `1.0`/`0.0` per detection (e.g. "was this a
    true positive at the matching tIoU threshold")."""
    if confidences.size == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.clip(np.digitize(confidences, bin_edges[1:-1]), 0, num_bins - 1)

    total = confidences.size
    ece = 0.0
    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        count = int(mask.sum())
        if count == 0:
            continue
        bin_confidence = float(confidences[mask].mean())
        bin_accuracy = float(correctness[mask].mean())
        ece += (count / total) * abs(bin_confidence - bin_accuracy)
    return round(ece, 6)


def brier_score(confidences: np.ndarray, correctness: np.ndarray) -> float:
    """Mean squared error between predicted confidence and binary
    correctness -- a simpler, bin-free calibration summary alongside ECE."""
    if confidences.size == 0:
        return 0.0
    return round(float(np.mean((confidences - correctness) ** 2)), 6)
