"""Perception pipeline (Prompt B): real pose/hand/yo-yo/tracking adapters,
kinematic feature computation, artefact I/O, evaluation, and overlay
rendering, on top of the `ml` package's replaceable-adapter architecture
(product principle #5; see `interfaces.py`/`adapters_registry.py`).

Importing this package registers every adapter defined here (`"mediapipe"`
pose/hand, `"pytorch"`/`"onnx"` yo-yo detectors, `"kalman"` tracker) with
`adapters_registry`, mirroring how importing `adapters_mock` registers the
`"mock"` adapters. None of these submodules import their optional heavy
dependency (mediapipe/torch/onnxruntime/opencv) at *module* import time --
only inside `__init__`/`predict`/etc. -- so importing `yoyovision_ml.perception`
itself never requires any of them to be installed; only actually
constructing/using a given adapter does.
"""

from __future__ import annotations

from yoyovision_ml.perception import (  # noqa: F401 -- imported for adapter registration
    detector_mediapipe,
    detector_onnx,
    detector_pytorch,
    tracker_kalman,
)

__all__ = [
    "detector_mediapipe",
    "detector_onnx",
    "detector_pytorch",
    "tracker_kalman",
]
