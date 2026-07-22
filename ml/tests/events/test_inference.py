from __future__ import annotations

import numpy as np
import pytest

from yoyovision_ml.domain import EQUIPMENT_EVENT_FAMILIES, EventFamily, FeatureFrame, FeatureSet
from yoyovision_ml.events.config import InferenceConfig, TrainingConfig
from yoyovision_ml.events.inference import run_inference
from yoyovision_ml.events.labels import NUM_CLASSES
from yoyovision_ml.events.model import build_model
from yoyovision_ml.events.windowing import NormalizationStats

torch = pytest.importorskip("torch")

_FEATURE_NAMES = ("feature_a", "feature_b")


def _features(num_frames: int = 10) -> FeatureSet:
    frames = tuple(
        FeatureFrame(frame_ms=i * 33, values={"feature_a": float(i), "feature_b": -float(i)})
        for i in range(num_frames)
    )
    return FeatureSet(frames=frames, feature_names=_FEATURE_NAMES, fps=30.0)


def _identity_normalization() -> NormalizationStats:
    return NormalizationStats(mean=np.zeros(2), std=np.ones(2))


def _model() -> object:
    torch.manual_seed(0)
    model = build_model(torch, input_dim=2, config=TrainingConfig(hidden_channels=4, num_blocks=1))
    model.eval()
    return model


def test_run_inference_returns_empty_list_for_a_clip_with_no_frames() -> None:
    empty_features = FeatureSet(frames=(), feature_names=_FEATURE_NAMES, fps=30.0)
    predictions = run_inference(
        torch,
        _model(),
        empty_features,
        _FEATURE_NAMES,
        _identity_normalization(),
        np.ones(NUM_CLASSES),
        "test-model",
        "v1",
        InferenceConfig(),
    )
    assert predictions == []


def test_run_inference_produces_one_prediction_per_class_when_everything_is_active() -> None:
    """`frame_activation_threshold=0.0` + `min_event_ms=0` forces every one of
    `NUM_CLASSES` per-frame probability columns to register as a single
    contiguous run over the whole clip, decoupling this test's expected
    count from the untrained model's actual (effectively random) logits."""
    predictions = run_inference(
        torch,
        _model(),
        _features(),
        _FEATURE_NAMES,
        _identity_normalization(),
        np.ones(NUM_CLASSES),
        "test-model",
        "v1",
        InferenceConfig(frame_activation_threshold=0.0, min_event_ms=0, uncertainty_threshold=0.0),
    )

    assert len(predictions) == NUM_CLASSES
    families = {prediction.family for prediction in predictions}
    assert len(families) == NUM_CLASSES


def test_run_inference_stamps_every_prediction_with_the_given_model_name_and_version() -> None:
    predictions = run_inference(
        torch,
        _model(),
        _features(),
        _FEATURE_NAMES,
        _identity_normalization(),
        np.ones(NUM_CLASSES),
        "trick-event-tcn",
        "v-42",
        InferenceConfig(frame_activation_threshold=0.0, min_event_ms=0, uncertainty_threshold=0.0),
    )

    assert predictions
    for prediction in predictions:
        assert prediction.model_name == "trick-event-tcn"
        assert prediction.model_version == "v-42"
        assert 0.0 <= prediction.confidence <= 1.0


def test_run_inference_relabels_predictions_as_unknown_at_max_uncertainty_threshold() -> None:
    predictions = run_inference(
        torch,
        _model(),
        _features(),
        _FEATURE_NAMES,
        _identity_normalization(),
        np.ones(NUM_CLASSES),
        "test-model",
        "v1",
        InferenceConfig(
            frame_activation_threshold=0.0,
            min_event_ms=0,
            uncertainty_threshold=1.0,
            uncertainty_action="relabel_unknown",
        ),
    )

    assert predictions
    for prediction in predictions:
        assert prediction.family == EventFamily.UNKNOWN_TECHNICAL_ELEMENT
        assert prediction.label == EventFamily.UNKNOWN_TECHNICAL_ELEMENT.value


def test_run_inference_flags_for_review_instead_of_relabeling_when_configured() -> None:
    predictions = run_inference(
        torch,
        _model(),
        _features(),
        _FEATURE_NAMES,
        _identity_normalization(),
        np.ones(NUM_CLASSES),
        "test-model",
        "v1",
        InferenceConfig(
            frame_activation_threshold=0.0,
            min_event_ms=0,
            uncertainty_threshold=1.0,
            uncertainty_action="flag_review",
        ),
    )

    assert predictions
    # `flag_review` must never rewrite `family`/`label` (unlike
    # `relabel_unknown`) -- every class's family, including the
    # `unknown_technical_element` class's own natural prediction, is
    # unchanged, and every prediction instead carries a review-flag note.
    predicted_families = {prediction.family for prediction in predictions}
    assert predicted_families == set(EventFamily) - EQUIPMENT_EVENT_FAMILIES
    for prediction in predictions:
        assert any("needs_review" in evidence.note for evidence in prediction.evidence)
