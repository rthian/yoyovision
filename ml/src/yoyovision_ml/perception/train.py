"""Training loop for the TinyYoyoNet yo-yo detector."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np
from yoyovision_ml.events.train import assert_no_leakage, player_grouped_split
from yoyovision_ml.perception._image_utils import hwc_uint8_to_chw_float01, resize_nearest
from yoyovision_ml.perception.checkpoint import DEFAULT_MODEL_NAME
from yoyovision_ml.perception.config import DetectorTrainingConfig
from yoyovision_ml.perception.evaluation import (
    PrecisionRecallResult,
    detector_precision_recall,
    ground_truth_from_dataset_track,
)
from yoyovision_ml.perception.model import INPUT_SIZE, build_tiny_yoyo_net, import_torch
from yoyovision_ml.perception.types import DetectorTrainingRunResult, DetectorTrainingSample

_MIN_IMPROVEMENT = 1e-4


def set_deterministic_seed(torch_module: Any, seed: int) -> None:
    torch_module.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _preprocess_image(image: np.ndarray, torch_module: Any) -> Any:
    resized = resize_nearest(np.asarray(image), (INPUT_SIZE, INPUT_SIZE))
    chw = hwc_uint8_to_chw_float01(resized)
    return torch_module.from_numpy(chw.copy()).float()


def _run_epoch(
    torch_module: Any,
    model: Any,
    samples: Sequence[DetectorTrainingSample],
    config: DetectorTrainingConfig,
    *,
    optimizer: Any | None,
    epoch_seed: int,
) -> float:
    rng = random.Random(epoch_seed)
    ordered = list(samples)
    if optimizer is not None:
        rng.shuffle(ordered)
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0
    smooth_l1 = torch_module.nn.SmoothL1Loss(reduction="none")
    bce = torch_module.nn.BCEWithLogitsLoss(reduction="none")

    with torch_module.set_grad_enabled(optimizer is not None):
        for start in range(0, len(ordered), config.batch_size):
            batch = ordered[start : start + config.batch_size]
            if not batch:
                continue
            images = torch_module.stack([_preprocess_image(sample.image, torch_module) for sample in batch])
            bbox_targets = torch_module.tensor(
                [list(sample.target_bbox) for sample in batch], dtype=torch_module.float32
            )
            visible_mask = torch_module.tensor(
                [sample.visible for sample in batch], dtype=torch_module.bool
            )
            outputs = model(images)
            bbox_preds = outputs[:, :4]
            confidence_logits = outputs[:, 4]

            bbox_loss = smooth_l1(bbox_preds, bbox_targets).mean(dim=1)
            bbox_loss = bbox_loss * visible_mask.float()
            confidence_targets = visible_mask.float()
            confidence_loss = bce(confidence_logits, confidence_targets)
            loss = (bbox_loss + confidence_loss).mean()

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_batches += 1

    return total_loss / max(total_batches, 1)


def _predictions_for_samples(
    torch_module: Any,
    model: Any,
    samples: Sequence[DetectorTrainingSample],
) -> list[Detection]:
    model.eval()
    detections: list[Detection] = []
    with torch_module.no_grad():
        for sample in samples:
            image = _preprocess_image(sample.image, torch_module).unsqueeze(0)
            output = model(image)[0]
            x, y, width, height, confidence_logit = output.tolist()
            confidence = 1.0 / (1.0 + pow(2.718281828459045, -confidence_logit))
            detections.append(
                Detection(
                    frame_ms=sample.frame_ms,
                    bbox=BoundingBox(
                        x=max(0.0, min(1.0, x)),
                        y=max(0.0, min(1.0, y)),
                        width=max(0.0, min(1.0, width)),
                        height=max(0.0, min(1.0, height)),
                    ),
                    confidence=confidence,
                    class_label="yoyo",
                    model_name=DEFAULT_MODEL_NAME,
                    model_version="eval",
                )
            )
    return detections


def _ground_truth_for_samples(samples: Sequence[DetectorTrainingSample]) -> list:
    class _Row:
        def __init__(self, sample: DetectorTrainingSample) -> None:
            self.frame_ms = sample.frame_ms
            self.visibility = "visible" if sample.visible else "fully_occluded"
            if sample.visible:
                x, y, width, height = sample.target_bbox
                self.point = type("P", (), {"x": x + width / 2, "y": y + height / 2})()
                self.bbox = type(
                    "B", (), {"x": x, "y": y, "width": width, "height": height}
                )()
            else:
                self.point = None
                self.bbox = None

    return ground_truth_from_dataset_track([_Row(sample) for sample in samples])


def _metrics_dict(result: PrecisionRecallResult, val_loss: float) -> dict[str, object]:
    return {
        "val_loss": val_loss,
        "precision": result.precision,
        "recall": result.recall,
        "true_positives": result.true_positives,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
    }


def _player_splits(
    train_samples: Sequence[DetectorTrainingSample],
    val_samples: Sequence[DetectorTrainingSample],
    test_samples: Sequence[DetectorTrainingSample],
) -> dict[str, str]:
    splits: dict[str, str] = {}
    for label, bucket in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        for sample in bucket:
            splits[sample.player_id] = label
    return splits


def train_detector(
    samples: Sequence[DetectorTrainingSample],
    config: DetectorTrainingConfig | None = None,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    training_data_source: str = "synthetic",
) -> DetectorTrainingRunResult:
    config = config or DetectorTrainingConfig()
    torch_module = import_torch()
    set_deterministic_seed(torch_module, config.seed)

    train_samples, val_samples, test_samples = player_grouped_split(
        samples, config.seed, config.train_ratio, config.val_ratio
    )
    assert_no_leakage(train_samples, val_samples, test_samples)
    if not train_samples or not val_samples:
        raise ValueError("Need non-empty train and val splits for yo-yo detector training.")

    model = build_tiny_yoyo_net(torch_module)
    optimizer = torch_module.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_val_loss = float("inf")
    best_state_dict: dict[str, Any] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    val_loss_history: list[float] = []

    for epoch in range(1, config.max_epochs + 1):
        _run_epoch(
            torch_module,
            model,
            train_samples,
            config,
            optimizer=optimizer,
            epoch_seed=config.seed + epoch,
        )
        val_loss = _run_epoch(
            torch_module,
            model,
            val_samples,
            config,
            optimizer=None,
            epoch_seed=config.seed,
        )
        val_loss_history.append(val_loss)
        if val_loss < best_val_loss - _MIN_IMPROVEMENT:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()

    val_predictions = _predictions_for_samples(torch_module, model, val_samples)
    val_ground_truth = _ground_truth_for_samples(val_samples)
    val_pr = detector_precision_recall(val_predictions, val_ground_truth)
    val_metrics = _metrics_dict(val_pr, best_val_loss)

    test_metrics: dict[str, object] | None = None
    if test_samples:
        test_predictions = _predictions_for_samples(torch_module, model, test_samples)
        test_ground_truth = _ground_truth_for_samples(test_samples)
        test_pr = detector_precision_recall(test_predictions, test_ground_truth)
        test_metrics = _metrics_dict(test_pr, best_val_loss)

    model_version = f"tiny-yoyo-seed{config.seed}-epoch{best_epoch}"
    return DetectorTrainingRunResult(
        torch_module=torch_module,
        model=model,
        config=config,
        model_name=model_name,
        model_version=model_version,
        player_splits=_player_splits(train_samples, val_samples, test_samples),
        best_epoch=best_epoch,
        val_loss_history=val_loss_history,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_sample_count=len(train_samples),
        val_sample_count=len(val_samples),
        test_sample_count=len(test_samples),
        training_data_source=training_data_source,
    )
