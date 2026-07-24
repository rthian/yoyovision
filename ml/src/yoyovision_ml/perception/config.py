"""Training configuration for the yo-yo detector."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectorTrainingConfig(BaseModel):
    seed: int = 42
    sample_fps: float = Field(default=15.0, gt=0.0)
    point_box_size: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Normalized width/height when only a centre point is annotated.",
    )
    frame_match_tolerance_ms: int = Field(
        default=0,
        ge=0,
        description="Max |annotated_ms - sampled_ms|; 0 means half the sample step.",
    )
    batch_size: int = Field(default=16, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    max_epochs: int = Field(default=30, gt=0)
    early_stopping_patience: int = Field(default=6, gt=0)
    train_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
