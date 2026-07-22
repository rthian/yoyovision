"""Temporal trick-event model (Prompt C): a trainable TCN that predicts
time-bounded 1A atomic trick events + outcomes from Prompt B perception
features, plus the always-available baselines it must be compared against.

Importing this package registers every `TemporalEventDetector` adapter
defined here (`"majority"`/`"rules"` from `baselines`, `"torch"` from
`detector_torch`) with `adapters_registry`, mirroring how importing
`yoyovision_ml.perception` registers its adapters. `detector_torch` never
imports `torch` at *module* import time -- only inside
`PyTorchTemporalEventDetector.__init__` -- so importing `yoyovision_ml.events`
itself never requires the optional `torch` extra to be installed; only
actually constructing/using the `"torch"` adapter does.
"""

from __future__ import annotations

from yoyovision_ml.events import (  # noqa: F401 -- imported for adapter registration
    baselines,
    detector_torch,
)

__all__ = ["baselines", "detector_torch"]
