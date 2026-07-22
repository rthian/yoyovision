"""Real (session-loading) ONNX Runtime adapter for the `YoyoDetector` protocol.

Mirrors `detector_pytorch.py`'s contract exactly (see that module's docstring
for the "no shipped weights, fail clearly" rationale): this adapter loads
whatever `.onnx` model file is configured and runs genuine inference through
`onnxruntime.InferenceSession`, but refuses to run un-configured.

Model I/O contract this adapter assumes (documented so a real exported model
can match it): one input named `input` of shape `(1, 3, 128, 128)` float32 in
`[0, 1]`, one output named `output` of shape `(1, 5)` -- `(x, y, width,
height, confidence_logit)`, matching `detector_pytorch.TinyYoyoNet`'s head so
a PyTorch checkpoint can be exported straight to ONNX for this adapter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from yoyovision_ml.adapters_registry import register_yoyo_detector
from yoyovision_ml.domain import BoundingBox, Detection
from yoyovision_ml.interfaces import FrameRef
from yoyovision_ml.perception._image_utils import hwc_uint8_to_chw_float01, resize_nearest
from yoyovision_ml.perception.errors import (
    MissingOptionalDependencyError,
    ModelWeightsNotConfiguredError,
)

ENV_MODEL_VAR = "YOYOVISION_ONNX_YOYO_MODEL"
_INPUT_SIZE = 128
_INPUT_NAME = "input"
_OUTPUT_NAME = "output"


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime
    except ImportError as exc:
        raise MissingOptionalDependencyError("onnxruntime", "onnx") from exc
    return onnxruntime


@register_yoyo_detector("onnx")
class ONNXYoyoDetector:
    """Real ONNX Runtime yo-yo detector adapter. Requires a configured `.onnx` model."""

    model_name = "onnx-yoyo-detector"

    def __init__(
        self, model_path: str | Path | None = None, providers: list[str] | None = None
    ) -> None:
        resolved = str(model_path) if model_path else os.environ.get(ENV_MODEL_VAR)
        if not resolved:
            raise ModelWeightsNotConfiguredError(
                "onnx",
                f"Pass model_path=... or set the {ENV_MODEL_VAR} environment "
                "variable to an exported .onnx model file.",
            )
        onnx_model_path = Path(resolved)
        if not onnx_model_path.exists():
            raise ModelWeightsNotConfiguredError(
                "onnx", f"Configured model file does not exist: {onnx_model_path}"
            )

        onnxruntime = _import_onnxruntime()
        self._session = onnxruntime.InferenceSession(
            str(onnx_model_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.model_version = f"onnx-model-hash-unset+ort{onnxruntime.__version__}"

    def predict(self, frame_batch: list[FrameRef]) -> list[Detection]:
        import numpy as np

        detections: list[Detection] = []
        for frame in frame_batch:
            if frame.array is None:
                continue
            input_tensor = self._preprocess(frame.array)
            (output,) = self._session.run([_OUTPUT_NAME], {_INPUT_NAME: input_tensor})
            x, y, width, height, confidence_logit = np.asarray(output).reshape(-1)[:5].tolist()
            confidence = 1.0 / (1.0 + pow(2.718281828459045, -confidence_logit))
            bbox = BoundingBox(
                x=max(0.0, min(1.0, x)),
                y=max(0.0, min(1.0, y)),
                width=max(0.0, min(1.0, width)),
                height=max(0.0, min(1.0, height)),
            )
            detections.append(
                Detection(
                    frame_ms=frame.frame_ms,
                    bbox=bbox,
                    confidence=confidence,
                    class_label="yoyo",
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
            )
        return detections

    def _preprocess(self, array: Any) -> Any:
        import numpy as np

        resized = resize_nearest(np.asarray(array), (_INPUT_SIZE, _INPUT_SIZE))
        chw = hwc_uint8_to_chw_float01(resized)
        return chw[np.newaxis, ...].astype(np.float32)
