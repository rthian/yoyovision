"""Real (checkpoint-loading) PyTorch adapter for the `TemporalEventDetector`
protocol -- Prompt C's trained model, as opposed to `baselines.py`'s
always-available, never-trained reference points.

No trained weights ship with this repository (same "no checkpoint ships"
caveat as `perception/detector_pytorch.py`). Construction fails immediately
and clearly (`ModelWeightsNotConfiguredError`) when no checkpoint is
configured -- it never fabricates a checkpoint or falls back to mock output.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from yoyovision_ml.adapters_registry import register_temporal_event_detector
from yoyovision_ml.domain import AnalysisEventPrediction, DeductionPrediction, FeatureSet
from yoyovision_ml.events.checkpoint import load_checkpoint
from yoyovision_ml.events.config import InferenceConfig
from yoyovision_ml.events.inference import run_inference
from yoyovision_ml.events.model import build_model, import_torch
from yoyovision_ml.events.windowing import NormalizationStats
from yoyovision_ml.perception.errors import ModelWeightsNotConfiguredError

ENV_WEIGHTS_VAR = "YOYOVISION_TORCH_EVENT_WEIGHTS"


@register_temporal_event_detector("torch")
class PyTorchTemporalEventDetector:
    """Real PyTorch temporal trick-event detector adapter. Requires a
    configured checkpoint (`.pt` weights + `.json` sidecar) written by
    `events.checkpoint.save_checkpoint` after an `events.train.train_model`
    run.

    Always returns an empty `DeductionPrediction` list -- Prompt C's 20
    classes exclude the 3 equipment-event families (see `labels.py`);
    equipment-event detection remains `adapters_mock.MockTemporalEventDetector`'s
    job (or a future dedicated model) until this package trains one.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        inference_config: InferenceConfig | None = None,
        device: str = "cpu",
    ) -> None:
        resolved = str(weights_path) if weights_path else os.environ.get(ENV_WEIGHTS_VAR)
        if not resolved:
            raise ModelWeightsNotConfiguredError(
                "torch",
                f"Pass weights_path=... or set the {ENV_WEIGHTS_VAR} environment "
                "variable to a .pt checkpoint written by events.checkpoint.save_checkpoint.",
            )
        checkpoint_path = Path(resolved)
        if not checkpoint_path.exists():
            raise ModelWeightsNotConfiguredError(
                "torch", f"Configured checkpoint does not exist: {checkpoint_path}"
            )

        torch_module = import_torch()
        state_dict, metadata = load_checkpoint(torch_module, checkpoint_path)

        self._torch: Any = torch_module
        self._device = device
        self._model = build_model(torch_module, metadata.input_dim, metadata.training_config)
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self._model.to(device)

        self._feature_names = metadata.feature_names
        self._normalization = NormalizationStats.from_dict(metadata.normalization)
        self._calibration_temperatures = np.array(metadata.calibration_temperatures)
        self._inference_config = inference_config or InferenceConfig()

        self.model_name = metadata.model_name
        self.model_version = f"{metadata.model_version}+torch{torch_module.__version__}"

    def predict(
        self, features: FeatureSet
    ) -> tuple[list[AnalysisEventPrediction], list[DeductionPrediction]]:
        events = run_inference(
            self._torch,
            self._model,
            features,
            self._feature_names,
            self._normalization,
            self._calibration_temperatures,
            self.model_name,
            self.model_version,
            self._inference_config,
            device=self._device,
        )
        return events, []
