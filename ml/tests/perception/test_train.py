from __future__ import annotations

from pathlib import Path

import pytest

from yoyovision_ml.perception.checkpoint import load_checkpoint
from yoyovision_ml.perception.config import DetectorTrainingConfig
from yoyovision_ml.perception.synthetic import generate_synthetic_detector_samples
from yoyovision_ml.perception.train import train_detector

torch = pytest.importorskip("torch")


def test_train_detector_smoke_on_synthetic_samples(tmp_path: Path) -> None:
    samples = generate_synthetic_detector_samples(
        seed=7, num_players=4, frames_per_player=6, image_size=48
    )
    result = train_detector(
        samples,
        config=DetectorTrainingConfig(seed=7, max_epochs=2, early_stopping_patience=1, batch_size=8),
        training_data_source="synthetic",
    )
    assert result.best_epoch >= 1
    assert result.val_metrics["precision"] >= 0.0
    assert result.training_data_source == "synthetic"

    from yoyovision_ml.perception.checkpoint import save_checkpoint, YoyoDetectorMetadata

    metadata = YoyoDetectorMetadata(
        model_name=result.model_name,
        model_version=result.model_version,
        training_config=result.config,
        player_splits=result.player_splits,
        best_epoch=result.best_epoch,
        val_loss_history=result.val_loss_history,
        val_metrics=result.val_metrics,
        test_metrics=result.test_metrics,
        train_sample_count=result.train_sample_count,
        val_sample_count=result.val_sample_count,
        test_sample_count=result.test_sample_count,
        torch_version=result.torch_module.__version__,
        training_data_source=result.training_data_source,
    )
    weights_path, _metadata_path = save_checkpoint(
        result.torch_module, result.model, metadata, tmp_path, "yoyo"
    )
    state_dict, loaded = load_checkpoint(torch, weights_path)
    assert loaded.training_data_source == "synthetic"
    assert state_dict
