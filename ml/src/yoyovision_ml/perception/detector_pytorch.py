"""Real (checkpoint-loading) PyTorch adapter for the `YoyoDetector` protocol.

No trained weights ship with this repository (see README's "Current model
status"). This adapter is the real, replaceable *runtime slot* Prompt B asks
for: it loads whatever checkpoint is configured and runs a genuine forward
pass, but it never fabricates a checkpoint or falls back to mock output --
if no weights are configured, construction fails immediately and clearly
(`ModelWeightsNotConfiguredError`) rather than producing meaningless
detections. Training a real checkpoint is Prompt C's job, not this one's.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from yoyovision_ml.adapters_registry import register_yoyo_detector
from yoyovision_ml.domain import BoundingBox, Detection
from yoyovision_ml.interfaces import FrameRef
from yoyovision_ml.perception._image_utils import hwc_uint8_to_chw_float01, resize_nearest
from yoyovision_ml.perception.model import INPUT_SIZE, build_tiny_yoyo_net
from yoyovision_ml.perception.errors import (
    MissingOptionalDependencyError,
    ModelWeightsNotConfiguredError,
)

ENV_WEIGHTS_VAR = "YOYOVISION_TORCH_YOYO_WEIGHTS"


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise MissingOptionalDependencyError("torch", "torch") from exc
    return torch


@register_yoyo_detector("pytorch")
class PyTorchYoyoDetector:
    """Real PyTorch yo-yo detector adapter. Requires a configured checkpoint."""

    model_name = "pytorch-yoyo-detector"

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str = "cpu",
        batch_size: int = 8,
    ) -> None:
        resolved = str(weights_path) if weights_path else os.environ.get(ENV_WEIGHTS_VAR)
        if not resolved:
            raise ModelWeightsNotConfiguredError(
                "pytorch",
                f"Pass weights_path=... or set the {ENV_WEIGHTS_VAR} environment "
                "variable to a .pt/.pth checkpoint.",
            )
        checkpoint_path = Path(resolved)
        if not checkpoint_path.exists():
            raise ModelWeightsNotConfiguredError(
                "pytorch", f"Configured checkpoint does not exist: {checkpoint_path}"
            )

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        torch = _import_torch()
        self._torch = torch
        self._device = device
        self._batch_size = batch_size
        self._model = build_tiny_yoyo_net(torch)
        # `weights_only=True` restricts unpickling to tensors/plain Python
        # containers -- a checkpoint file is untrusted input (may have been
        # uploaded or shared), so we never allow arbitrary object unpickling.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self._model.to(device)

        checkpoint_version = (
            checkpoint.get("model_version", "unconfigured")
            if isinstance(checkpoint, dict)
            else "unconfigured"
        )
        self.model_version = f"{checkpoint_version}+torch{torch.__version__}"

    def predict(self, frame_batch: list[FrameRef]) -> list[Detection]:
        """Runs inference in chunks of `self._batch_size` frames per forward
        pass (Prompt F: "Add bounded batching") instead of one frame at a
        time -- bounded so a very long clip never builds one unbounded
        batch tensor regardless of how many frames are passed in."""
        torch = self._torch
        detections: list[Detection] = []
        usable_frames = [frame for frame in frame_batch if frame.array is not None]

        with torch.no_grad():
            for start in range(0, len(usable_frames), self._batch_size):
                chunk = usable_frames[start : start + self._batch_size]
                tensors = torch.stack([self._preprocess(frame.array) for frame in chunk])
                outputs = self._model(tensors.to(self._device))
                for frame, output in zip(chunk, outputs, strict=True):
                    x, y, width, height, confidence_logit = output.tolist()
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

        resized = resize_nearest(np.asarray(array), (INPUT_SIZE, INPUT_SIZE))
        chw = hwc_uint8_to_chw_float01(resized)
        return self._torch.from_numpy(chw.copy()).float()
