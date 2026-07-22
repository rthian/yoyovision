"""Experiment/training/inference configuration for the temporal trick-event model.

Every training run's `TrainingConfig` is saved verbatim into the checkpoint's
JSON metadata sidecar (see `events/train.py`, `events/artifact.py`) -- per
Prompt C's "experiment configuration saved with each run" requirement. Plain
Pydantic models (not dataclasses) so they round-trip through JSON without a
hand-written encoder, matching the `perception.artifact.PerceptionMetadata`
convention already used elsewhere in this package.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FeatureSubsetName = Literal[
    "fused", "skeleton", "trajectory", "kinematics_only", "multimodal_fused"
]
NmsStrategy = Literal["merge", "suppress"]
UncertaintyAction = Literal["relabel_unknown", "flag_review"]


class TrainingConfig(BaseModel):
    """Everything needed to reproduce one training run byte-for-byte given
    the same input samples (Prompt C: "deterministic seed", "configurable
    temporal window", "experiment configuration saved with each run")."""

    #: Which Prompt-B feature columns the model consumes -- also drives the
    #: "skeleton-only" / "yo-yo-trajectory-only" / "fused" ablation baselines.
    feature_subset: FeatureSubsetName = "fused"

    #: Fixed-length training window, in milliseconds, and the stride between
    #: consecutive windows sampled from one clip.
    window_ms: int = Field(default=4000, gt=0)
    stride_ms: int = Field(default=2000, gt=0)

    seed: int = 42

    #: TCN encoder hyperparameters -- kept modest per Prompt C's "Implement a
    #: modest and reproducible baseline before using a large video transformer."
    hidden_channels: int = Field(default=64, gt=0)
    num_blocks: int = Field(default=4, gt=0)
    kernel_size: int = Field(default=3, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    batch_size: int = Field(default=16, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    max_epochs: int = Field(default=50, gt=0)
    early_stopping_patience: int = Field(default=8, gt=0)

    #: Class-balanced loss weighting strategy for the multi-label
    #: classification head (Prompt C: "class-balanced sampling or loss
    #: weighting"). `"inverse_frequency"` sets each class's BCE `pos_weight`
    #: to `n_negative / n_positive` (clipped) computed on the *training*
    #: split only, so validation/test never leak into the weighting.
    class_balance_strategy: Literal["none", "inverse_frequency"] = "inverse_frequency"
    max_pos_weight: float = Field(default=50.0, gt=0.0)

    #: Split ratios for the player-grouped train/val/test split (Prompt C:
    #: "player-grouped data splits", "no train/test leakage").
    train_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    # test_ratio is implied as 1 - train_ratio - val_ratio.


class InferenceConfig(BaseModel):
    """Decode-time knobs, independent of how the model was trained -- these
    can be tuned per deployment without retraining."""

    #: Per-frame probability threshold above which a class is considered
    #: "active" at that frame (before grouping into discrete events).
    frame_activation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    #: Discard any candidate event shorter than this (denoise single-frame
    #: spikes before they become spurious detections).
    min_event_ms: int = Field(default=120, ge=0)

    #: Temporal non-maximum-suppression / merge behaviour for overlapping or
    #: adjacent same-class candidate events (Prompt C: "configurable temporal
    #: non-maximum suppression or event merging").
    nms_strategy: NmsStrategy = "merge"
    nms_iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    merge_gap_ms: int = Field(default=150, ge=0)

    #: Below this *calibrated* confidence, apply `uncertainty_action` (Prompt
    #: C: "an uncertainty threshold that converts low-confidence predictions
    #: to unknown_technical_element or sends them to human review").
    uncertainty_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    uncertainty_action: UncertaintyAction = "relabel_unknown"

    #: tIoU thresholds `metrics.temporal_map` reports AP at (Prompt C:
    #: "temporal mAP at configurable tIoU thresholds").
    tiou_thresholds: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)

    #: Calibration-error bin count for `metrics.confidence_calibration`.
    calibration_bins: int = Field(default=10, gt=1)
