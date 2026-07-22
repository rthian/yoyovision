from __future__ import annotations

from pathlib import Path

import pytest

from yoyovision_ml.adapters_registry import create_temporal_event_detector
from yoyovision_ml.domain import FeatureFrame, FeatureSet
from yoyovision_ml.events.checkpoint import EventModelMetadata, save_checkpoint
from yoyovision_ml.events.config import InferenceConfig, TrainingConfig
from yoyovision_ml.events.detector_torch import ENV_WEIGHTS_VAR, PyTorchTemporalEventDetector
from yoyovision_ml.events.labels import NUM_CLASSES
from yoyovision_ml.events.model import build_model
from yoyovision_ml.perception.errors import ModelWeightsNotConfiguredError

torch = pytest.importorskip("torch")

_FEATURE_NAMES = ("feature_a", "feature_b")


def _write_checkpoint(tmp_path: Path, model_version: str = "v-1") -> Path:
    torch.manual_seed(0)
    config = TrainingConfig(hidden_channels=4, num_blocks=1)
    model = build_model(torch, input_dim=2, config=config)
    metadata = EventModelMetadata(
        model_name="trick-event-tcn",
        model_version=model_version,
        training_config=config,
        feature_names=_FEATURE_NAMES,
        input_dim=2,
        normalization={"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        calibration_temperatures=[1.0] * NUM_CLASSES,
        player_splits={"player-a": "train"},
        best_epoch=1,
        val_loss_history=[1.0],
        val_metrics={},
        test_metrics=None,
        train_sample_count=1,
        val_sample_count=1,
        test_sample_count=1,
        torch_version=torch.__version__,
    )
    weights_path, _ = save_checkpoint(torch, model, metadata, tmp_path, "checkpoint")
    return weights_path


def _features(num_frames: int = 10) -> FeatureSet:
    frames = tuple(
        FeatureFrame(frame_ms=i * 33, values={"feature_a": float(i), "feature_b": -float(i)})
        for i in range(num_frames)
    )
    return FeatureSet(frames=frames, feature_names=_FEATURE_NAMES, fps=30.0)


def test_construction_raises_clearly_when_no_checkpoint_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_WEIGHTS_VAR, raising=False)
    with pytest.raises(ModelWeightsNotConfiguredError):
        PyTorchTemporalEventDetector()


def test_construction_raises_clearly_when_the_configured_checkpoint_is_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelWeightsNotConfiguredError):
        PyTorchTemporalEventDetector(weights_path=tmp_path / "does-not-exist.pt")


def test_construction_reads_the_checkpoint_path_from_the_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights_path = _write_checkpoint(tmp_path)
    monkeypatch.setenv(ENV_WEIGHTS_VAR, str(weights_path))

    detector = PyTorchTemporalEventDetector()
    assert detector.model_name == "trick-event-tcn"


def test_model_version_embeds_checkpoint_version_and_installed_torch_version(
    tmp_path: Path,
) -> None:
    weights_path = _write_checkpoint(tmp_path, model_version="v-42")
    detector = PyTorchTemporalEventDetector(weights_path=weights_path)
    assert detector.model_version == f"v-42+torch{torch.__version__}"


def test_predict_returns_predictions_and_always_an_empty_deduction_list(tmp_path: Path) -> None:
    weights_path = _write_checkpoint(tmp_path)
    detector = PyTorchTemporalEventDetector(
        weights_path=weights_path,
        inference_config=InferenceConfig(frame_activation_threshold=0.0, min_event_ms=0),
    )

    predictions, deductions = detector.predict(_features())

    assert deductions == []
    assert isinstance(predictions, list)
    for prediction in predictions:
        assert prediction.model_name == "trick-event-tcn"


def test_predict_returns_empty_predictions_for_a_clip_with_no_frames(tmp_path: Path) -> None:
    weights_path = _write_checkpoint(tmp_path)
    detector = PyTorchTemporalEventDetector(weights_path=weights_path)
    empty_features = FeatureSet(frames=(), feature_names=_FEATURE_NAMES, fps=30.0)

    predictions, deductions = detector.predict(empty_features)

    assert predictions == []
    assert deductions == []


def test_torch_detector_is_registered_under_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights_path = _write_checkpoint(tmp_path)
    monkeypatch.setenv(ENV_WEIGHTS_VAR, str(weights_path))

    detector = create_temporal_event_detector("torch")
    assert isinstance(detector, PyTorchTemporalEventDetector)
