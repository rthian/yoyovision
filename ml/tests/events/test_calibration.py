from __future__ import annotations

import numpy as np
import pytest

from yoyovision_ml.events.calibration import (
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    fit_temperature_per_class,
    sigmoid,
)


def test_sigmoid_of_zero_is_one_half() -> None:
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_sigmoid_is_monotonically_increasing() -> None:
    values = sigmoid(np.array([-5.0, -1.0, 0.0, 1.0, 5.0]))
    assert np.all(np.diff(values) > 0)


def test_sigmoid_saturates_towards_zero_and_one() -> None:
    values = sigmoid(np.array([-50.0, 50.0]))
    assert values[0] == pytest.approx(0.0, abs=1e-6)
    assert values[1] == pytest.approx(1.0, abs=1e-6)


def test_fit_temperature_returns_one_for_empty_logits() -> None:
    assert fit_temperature(np.array([]), np.array([])) == 1.0


def test_fit_temperature_softens_grossly_overconfident_logits() -> None:
    """Logits saying "100% confident" but actually wrong half the time should
    fit a temperature > 1 (softening), since the raw sigmoid(logits) would be
    badly miscalibrated otherwise."""
    logits = np.array([10.0] * 50 + [10.0] * 50)
    labels = np.array([1.0] * 50 + [0.0] * 50)  # only right half the time
    temperature = fit_temperature(logits, labels)
    assert temperature > 1.0


def test_fit_temperature_per_class_returns_one_temperature_per_column() -> None:
    logits = np.random.default_rng(0).normal(size=(20, 3))
    labels = (logits > 0).astype(np.float64)
    temperatures = fit_temperature_per_class(logits, labels)
    assert temperatures.shape == (3,)


def test_apply_temperature_with_scalar_temperature_matches_manual_sigmoid() -> None:
    logits = np.array([1.0, -1.0, 2.0])
    calibrated = apply_temperature(logits, 2.0)
    assert np.allclose(calibrated, sigmoid(logits / 2.0))


def test_apply_temperature_broadcasts_per_class_temperature() -> None:
    logits = np.array([[1.0, 2.0], [3.0, 4.0]])
    temperatures = np.array([1.0, 2.0])
    calibrated = apply_temperature(logits, temperatures)
    assert np.allclose(calibrated, sigmoid(logits / temperatures))


def test_expected_calibration_error_is_zero_for_empty_input() -> None:
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0


def test_expected_calibration_error_is_near_zero_when_confidence_matches_accuracy() -> None:
    confidences = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    correctness = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])  # 9/10 = 0.9
    assert expected_calibration_error(confidences, correctness) < 0.05


def test_expected_calibration_error_is_high_when_overconfident_and_always_wrong() -> None:
    confidences = np.full(10, 0.95)
    correctness = np.zeros(10)
    ece = expected_calibration_error(confidences, correctness)
    assert ece == pytest.approx(0.95, abs=0.01)


def test_brier_score_is_zero_for_empty_input() -> None:
    assert brier_score(np.array([]), np.array([])) == 0.0


def test_brier_score_is_zero_for_perfect_confidence() -> None:
    confidences = np.array([1.0, 0.0, 1.0])
    correctness = np.array([1.0, 0.0, 1.0])
    assert brier_score(confidences, correctness) == 0.0


def test_brier_score_is_one_for_maximally_wrong_confidence() -> None:
    confidences = np.array([1.0, 0.0])
    correctness = np.array([0.0, 1.0])
    assert brier_score(confidences, correctness) == 1.0
