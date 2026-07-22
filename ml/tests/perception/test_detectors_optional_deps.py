"""Tests for the real (optional-dependency) detector/pose/hand adapters.

`torch`, `onnxruntime`, `mediapipe`, and `cv2` are all confirmed absent from
this test environment, so these tests exercise the required "fail clearly,
never silently fall back to mock output" behaviour -- the actual forward-pass
code paths need a real checkpoint/model file and are exercised manually once
weights are trained (Prompt C), per the project's "never fabricate a
checkpoint" rule (see `detector_pytorch.py`'s docstring).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from yoyovision_ml.perception._image_utils import hwc_uint8_to_chw_float01, resize_nearest
from yoyovision_ml.perception.errors import (
    MissingOptionalDependencyError,
    ModelWeightsNotConfiguredError,
)


# --------------------------------------------------------------------------- #
# _image_utils (no optional dependency required)
# --------------------------------------------------------------------------- #
def test_resize_nearest_changes_shape_to_requested_size() -> None:
    array = np.zeros((64, 48, 3), dtype=np.uint8)
    resized = resize_nearest(array, (128, 128))
    assert resized.shape == (128, 128, 3)


def test_resize_nearest_preserves_pixel_values_for_downscale() -> None:
    array = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    resized = resize_nearest(array, (8, 8))
    assert resized.shape == (8, 8, 3)
    assert resized.dtype == np.uint8


def test_hwc_uint8_to_chw_float01_transposes_and_normalizes() -> None:
    array = np.full((4, 5, 3), 255, dtype=np.uint8)
    chw = hwc_uint8_to_chw_float01(array)
    assert chw.shape == (3, 4, 5)
    assert chw.dtype == np.float32
    assert np.allclose(chw, 1.0)


def test_hwc_uint8_to_chw_float01_zero_stays_zero() -> None:
    array = np.zeros((2, 2, 3), dtype=np.uint8)
    chw = hwc_uint8_to_chw_float01(array)
    assert np.allclose(chw, 0.0)


# --------------------------------------------------------------------------- #
# PyTorchYoyoDetector
# --------------------------------------------------------------------------- #
def test_pytorch_detector_without_weights_raises_model_weights_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yoyovision_ml.perception.detector_pytorch import ENV_WEIGHTS_VAR, PyTorchYoyoDetector

    monkeypatch.delenv(ENV_WEIGHTS_VAR, raising=False)
    with pytest.raises(ModelWeightsNotConfiguredError):
        PyTorchYoyoDetector(weights_path=None)


def test_pytorch_detector_with_nonexistent_checkpoint_path_raises() -> None:
    from yoyovision_ml.perception.detector_pytorch import PyTorchYoyoDetector

    with pytest.raises(ModelWeightsNotConfiguredError):
        PyTorchYoyoDetector(weights_path="/nonexistent/checkpoint.pt")


def test_pytorch_detector_with_existing_checkpoint_but_missing_torch_raises_dependency_error(
    tmp_path: Path,
) -> None:
    """`torch` is confirmed absent, so once weights_path resolves to a real
    file, the next failure must be the *dependency* error, not weights."""
    from yoyovision_ml.perception.detector_pytorch import PyTorchYoyoDetector

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"not a real checkpoint")

    with pytest.raises(MissingOptionalDependencyError) as exc_info:
        PyTorchYoyoDetector(weights_path=checkpoint)
    assert "torch" in str(exc_info.value)


def test_pytorch_env_var_is_used_when_weights_path_not_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from yoyovision_ml.perception.detector_pytorch import ENV_WEIGHTS_VAR, PyTorchYoyoDetector

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    monkeypatch.setenv(ENV_WEIGHTS_VAR, str(checkpoint))

    # Should get past the "not configured" check and fail on the missing
    # `torch` dependency instead.
    with pytest.raises(MissingOptionalDependencyError):
        PyTorchYoyoDetector(weights_path=None)


# --------------------------------------------------------------------------- #
# ONNXYoyoDetector
# --------------------------------------------------------------------------- #
def test_onnx_detector_without_model_path_raises_model_weights_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yoyovision_ml.perception.detector_onnx import ENV_MODEL_VAR, ONNXYoyoDetector

    monkeypatch.delenv(ENV_MODEL_VAR, raising=False)
    with pytest.raises(ModelWeightsNotConfiguredError):
        ONNXYoyoDetector(model_path=None)


def test_onnx_detector_with_nonexistent_model_path_raises() -> None:
    from yoyovision_ml.perception.detector_onnx import ONNXYoyoDetector

    with pytest.raises(ModelWeightsNotConfiguredError):
        ONNXYoyoDetector(model_path="/nonexistent/model.onnx")


def test_onnx_detector_with_existing_model_but_missing_onnxruntime_raises_dependency_error(
    tmp_path: Path,
) -> None:
    from yoyovision_ml.perception.detector_onnx import ONNXYoyoDetector

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"not a real onnx model")

    with pytest.raises(MissingOptionalDependencyError) as exc_info:
        ONNXYoyoDetector(model_path=model_path)
    assert "onnxruntime" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# MediaPipe pose/hand estimators
# --------------------------------------------------------------------------- #
def test_mediapipe_pose_estimator_raises_missing_dependency_without_mediapipe() -> None:
    from yoyovision_ml.perception.detector_mediapipe import MediaPipePoseEstimator

    with pytest.raises(MissingOptionalDependencyError) as exc_info:
        MediaPipePoseEstimator()
    assert "mediapipe" in str(exc_info.value)


def test_mediapipe_hand_estimator_raises_missing_dependency_without_mediapipe() -> None:
    from yoyovision_ml.perception.detector_mediapipe import MediaPipeHandEstimator

    with pytest.raises(MissingOptionalDependencyError) as exc_info:
        MediaPipeHandEstimator()
    assert "mediapipe" in str(exc_info.value)
