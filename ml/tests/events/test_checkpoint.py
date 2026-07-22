from __future__ import annotations

from pathlib import Path

import pytest

from yoyovision_ml.events.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    EventModelMetadata,
    load_checkpoint,
    save_checkpoint,
)
from yoyovision_ml.events.config import TrainingConfig

torch = pytest.importorskip("torch")


def _metadata(**overrides: object) -> EventModelMetadata:
    defaults: dict[str, object] = {
        "model_name": "trick-event-tcn",
        "model_version": "test-1",
        "training_config": TrainingConfig(),
        "feature_names": ("yoyo_speed", "hand_distance"),
        "input_dim": 2,
        "normalization": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        "calibration_temperatures": [1.0] * 20,
        "player_splits": {"player-a": "train", "player-b": "val"},
        "best_epoch": 3,
        "val_loss_history": [1.0, 0.8, 0.75],
        "val_metrics": {"macro_f1": 0.5},
        "test_metrics": None,
        "train_sample_count": 10,
        "val_sample_count": 3,
        "test_sample_count": 2,
        "torch_version": "test",
    }
    defaults.update(overrides)
    return EventModelMetadata(**defaults)  # type: ignore[arg-type]


def test_save_checkpoint_writes_a_pt_file_and_a_json_sidecar(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    weights_path, metadata_path = save_checkpoint(torch, model, _metadata(), tmp_path, "model")

    assert weights_path == tmp_path / "model.pt"
    assert metadata_path == tmp_path / "model.json"
    assert weights_path.exists()
    assert metadata_path.exists()


def test_load_checkpoint_round_trips_state_dict_and_metadata(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    metadata = _metadata(model_version="v-round-trip")
    weights_path, _ = save_checkpoint(torch, model, metadata, tmp_path, "model")

    state_dict, loaded_metadata = load_checkpoint(torch, weights_path)

    assert loaded_metadata.model_version == "v-round-trip"
    assert loaded_metadata.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert set(state_dict) == set(model.state_dict())
    for key, tensor in model.state_dict().items():
        assert torch.equal(state_dict[key], tensor)


def test_load_checkpoint_restores_a_model_that_produces_identical_output(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 3)
    weights_path, _ = save_checkpoint(torch, model, _metadata(input_dim=4), tmp_path, "model")

    rebuilt = torch.nn.Linear(4, 3)
    state_dict, _ = load_checkpoint(torch, weights_path)
    rebuilt.load_state_dict(state_dict)

    sample_input = torch.randn(1, 4)
    with torch.no_grad():
        assert torch.equal(model(sample_input), rebuilt(sample_input))


def test_metadata_round_trips_through_json_preserving_training_config(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    metadata = _metadata(training_config=TrainingConfig(seed=123, hidden_channels=32))
    weights_path, _ = save_checkpoint(torch, model, metadata, tmp_path, "model")

    _, loaded_metadata = load_checkpoint(torch, weights_path)

    assert loaded_metadata.training_config.seed == 123
    assert loaded_metadata.training_config.hidden_channels == 32


def test_metadata_defaults_training_data_source_to_synthetic() -> None:
    assert _metadata().training_data_source == "synthetic"


def test_metadata_test_metrics_defaults_to_none_until_a_test_set_is_evaluated() -> None:
    assert _metadata().test_metrics is None
    populated = _metadata(test_metrics={"macro_f1": 0.42})
    assert populated.test_metrics == {"macro_f1": 0.42}
