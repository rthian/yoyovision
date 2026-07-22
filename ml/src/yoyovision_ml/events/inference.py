"""Shared model-forward-pass -> `AnalysisEventPrediction` pipeline.

The one place that ties together `windowing` (feature matrix + normalization),
a trained `model.py` TCN, `calibration` (temperature scaling), `decode`
(thresholding/NMS/uncertainty), and `convert` (domain conversion) into a
single inference call -- used identically by `train.py`'s post-training
validation/test reporting and by `detector_torch.py`'s
`TemporalEventDetector.predict`, so there is exactly one code path from
"trained model + raw features" to "persisted `AnalysisEventPrediction`s",
never two subtly different ones.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yoyovision_ml.domain import AnalysisEventPrediction, FeatureSet
from yoyovision_ml.events.calibration import apply_temperature, sigmoid
from yoyovision_ml.events.config import InferenceConfig
from yoyovision_ml.events.convert import to_analysis_event_prediction
from yoyovision_ml.events.decode import decode_predictions
from yoyovision_ml.events.windowing import NormalizationStats, feature_matrix, frame_timestamps_ms


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    result: np.ndarray = exp / exp.sum(axis=-1, keepdims=True)
    return result


def run_inference(
    torch_module: Any,
    model: Any,
    features: FeatureSet,
    feature_names: tuple[str, ...],
    normalization: NormalizationStats,
    calibration_temperatures: np.ndarray,
    model_name: str,
    model_version: str,
    inference_config: InferenceConfig,
    device: str = "cpu",
) -> list[AnalysisEventPrediction]:
    """Runs `model` on one clip's `features` end to end and returns decoded,
    calibrated `AnalysisEventPrediction`s. `model` must already be in
    `eval()` mode and on `device` -- this function never toggles either
    (`train.py` sets `.eval()` around its validation/test passes;
    `detector_torch.py` sets it once at load time)."""
    if not features.frames:
        return []

    frame_ms = frame_timestamps_ms(features)
    matrix = normalization.apply(feature_matrix(features, feature_names))

    with torch_module.no_grad():
        tensor = torch_module.from_numpy(matrix).float().unsqueeze(0).to(device)
        outputs = model(tensor)

    class_logits = outputs["class_logits"].squeeze(0).cpu().numpy()
    start_logits = outputs["start_logits"].squeeze(0).cpu().numpy()
    end_logits = outputs["end_logits"].squeeze(0).cpu().numpy()
    outcome_logits = outputs["outcome_logits"].squeeze(0).cpu().numpy()

    class_probs = apply_temperature(class_logits, calibration_temperatures)
    start_probs = sigmoid(start_logits)
    end_probs = sigmoid(end_logits)
    outcome_probs = _softmax(outcome_logits)

    detections = decode_predictions(
        frame_ms,
        class_probs,
        outcome_probs,
        model_version,
        inference_config,
        start_probs,
        end_probs,
    )
    return [to_analysis_event_prediction(detection, model_name) for detection in detections]
