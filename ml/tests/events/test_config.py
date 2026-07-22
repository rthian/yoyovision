from __future__ import annotations

import pytest
from pydantic import ValidationError

from yoyovision_ml.events.config import InferenceConfig, TrainingConfig


def test_training_config_defaults_are_valid() -> None:
    config = TrainingConfig()
    assert config.feature_subset == "fused"
    assert config.window_ms == 4000
    assert config.stride_ms == 2000
    assert config.seed == 42
    assert config.class_balance_strategy == "inverse_frequency"
    assert config.train_ratio + config.val_ratio < 1.0


def test_training_config_rejects_non_positive_window_ms() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(window_ms=0)


def test_training_config_rejects_dropout_out_of_bounds() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(dropout=1.0)
    with pytest.raises(ValidationError):
        TrainingConfig(dropout=-0.1)


def test_training_config_accepts_valid_feature_subset_literals() -> None:
    for subset in ("fused", "skeleton", "trajectory"):
        config = TrainingConfig(feature_subset=subset)
        assert config.feature_subset == subset


def test_training_config_rejects_invalid_feature_subset() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(feature_subset="not-a-real-subset")  # type: ignore[arg-type]


def test_training_config_round_trips_through_json() -> None:
    config = TrainingConfig(seed=7, hidden_channels=16, num_blocks=2)
    restored = TrainingConfig.model_validate_json(config.model_dump_json())
    assert restored == config


def test_training_config_model_copy_update_overrides_single_field() -> None:
    config = TrainingConfig()
    updated = config.model_copy(update={"feature_subset": "skeleton"})
    assert updated.feature_subset == "skeleton"
    assert updated.seed == config.seed


def test_inference_config_defaults_are_valid() -> None:
    config = InferenceConfig()
    assert config.frame_activation_threshold == 0.5
    assert config.nms_strategy == "merge"
    assert config.uncertainty_action == "relabel_unknown"
    assert config.tiou_thresholds == (0.1, 0.3, 0.5, 0.7, 0.9)


def test_inference_config_rejects_threshold_outside_zero_one() -> None:
    with pytest.raises(ValidationError):
        InferenceConfig(frame_activation_threshold=1.5)
    with pytest.raises(ValidationError):
        InferenceConfig(uncertainty_threshold=-0.1)


def test_inference_config_accepts_suppress_nms_strategy() -> None:
    config = InferenceConfig(nms_strategy="suppress")
    assert config.nms_strategy == "suppress"


def test_inference_config_accepts_flag_review_uncertainty_action() -> None:
    config = InferenceConfig(uncertainty_action="flag_review")
    assert config.uncertainty_action == "flag_review"


def test_inference_config_round_trips_through_json() -> None:
    config = InferenceConfig(uncertainty_threshold=0.6, tiou_thresholds=(0.25, 0.75))
    restored = InferenceConfig.model_validate_json(config.model_dump_json())
    assert restored == config
