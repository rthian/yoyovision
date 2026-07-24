"""Training sample types for the yo-yo detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from yoyovision_ml.perception.config import DetectorTrainingConfig


@dataclass(frozen=True, slots=True)
class DetectorTrainingSample:
    video_id: str
    player_id: str
    frame_ms: int
    image: np.ndarray
    target_bbox: tuple[float, float, float, float]
    visible: bool


@dataclass(frozen=True, slots=True)
class DetectorTrainingRunResult:
    torch_module: Any
    model: Any
    config: DetectorTrainingConfig
    model_name: str
    model_version: str
    player_splits: dict[str, str]
    best_epoch: int
    val_loss_history: list[float]
    val_metrics: dict[str, object]
    test_metrics: dict[str, object] | None
    train_sample_count: int
    val_sample_count: int
    test_sample_count: int
    training_data_source: str
