"""Full training loop for the temporal trick-event TCN, per Prompt C's
"TRAINING REQUIREMENTS": player-grouped splits with a leakage check,
class-balanced loss weighting, a configurable temporal window (via
`windowing.slice_windows`, already driven by `TrainingConfig`), a
deterministic seed, early stopping, checkpoint metadata, and experiment
configuration saved with each run (`checkpoint.EventModelMetadata`).

Prompt C also asks for "metrics by class and player skill band where
available": `metrics.EvaluationReport` already reports per-class F1, but no
player skill-band field exists anywhere in `dataset.schema`/`domain` today,
so skill-band breakdowns are simply not available yet -- "where available"
covers that omission rather than this module fabricating a band.

Variable-length windows (see `windowing.slice_windows`'s docstring -- clips
have slightly different frame counts even at the same `window_ms`) are
processed one at a time; `TrainingConfig.batch_size` is honoured as a
gradient-*accumulation* count rather than a padded tensor batch dimension.
This is a deliberate simplification appropriate to Prompt C's "modest and
reproducible baseline" -- not a claim about how a production trainer should
batch variable-length sequences.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from yoyovision_ml.domain import AnalysisEventPrediction, DeductionPrediction, FeatureSet
from yoyovision_ml.events.calibration import fit_temperature_per_class
from yoyovision_ml.events.config import InferenceConfig, TrainingConfig
from yoyovision_ml.events.inference import run_inference
from yoyovision_ml.events.labels import FEATURE_SUBSETS, NUM_CLASSES, NUM_OUTCOMES
from yoyovision_ml.events.metrics import EvaluationReport, evaluate_detector
from yoyovision_ml.events.model import build_model, import_torch
from yoyovision_ml.events.types import TrainingSample
from yoyovision_ml.events.windowing import (
    NormalizationStats,
    Window,
    build_frame_targets,
    feature_matrix,
    fit_normalization,
    frame_timestamps_ms,
    slice_windows,
)

DEFAULT_MODEL_NAME = "trick-event-tcn"

#: Fixed BCE `pos_weight` for the boundary (start/end) heads. Boundary
#: targets are exactly one positive frame per event per class -- far sparser
#: than class-activation targets, so `TrainingConfig.class_balance_strategy`
#: (tuned for the classification head) does not apply here. A single fixed
#: constant, not user-configurable, keeps this a simple, documented default
#: rather than another knob to get wrong.
_BOUNDARY_POS_WEIGHT = 25.0

#: Minimum validation-loss improvement to reset early-stopping patience.
_MIN_IMPROVEMENT = 1e-4


def player_grouped_split(
    samples: Sequence[TrainingSample],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[TrainingSample], list[TrainingSample], list[TrainingSample]]:
    """Deterministically splits `samples` into `(train, val, test)` by
    `player_id`, so no player's clips ever straddle two splits (Prompt C:
    "player-grouped data splits", "no train/test leakage").

    Same algorithm as `dataset.splits.generate_player_grouped_splits` (sort
    player ids, seeded shuffle, greedily assign each player to whichever
    split is furthest below its target share) but implemented directly
    against `TrainingSample.player_id` rather than requiring a full
    `DatasetVideo`, since Prompt C training samples may be entirely
    synthetic (`events/synthetic.py`) and never touch the `dataset`
    package's schema at all.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must leave a "
            "positive test_ratio"
        )
    ratios: dict[str, float] = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

    samples_by_player: dict[str, list[TrainingSample]] = defaultdict(list)
    for sample in samples:
        samples_by_player[sample.player_id].append(sample)

    player_ids = sorted(samples_by_player)
    rng = random.Random(seed)
    rng.shuffle(player_ids)

    total = len(samples)
    target_counts = {split: round(ratio * total) for split, ratio in ratios.items()}
    running_counts: dict[str, int] = dict.fromkeys(ratios, 0)
    split_order = sorted(ratios, key=lambda s: -ratios[s])

    player_splits: dict[str, str] = {}
    for player_id in player_ids:
        count = len(samples_by_player[player_id])
        best_split = min(
            split_order,
            key=lambda s: (
                (running_counts[s] / target_counts[s]) if target_counts[s] > 0 else float("inf")
            ),
        )
        player_splits[player_id] = best_split
        running_counts[best_split] += count

    buckets: dict[str, list[TrainingSample]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        buckets[player_splits[sample.player_id]].append(sample)
    return buckets["train"], buckets["val"], buckets["test"]


def assert_no_leakage(*splits: Sequence[TrainingSample]) -> None:
    """Raises `ValueError` if any `player_id` appears in more than one of
    `splits` -- an independent correctness check on top of
    `player_grouped_split`'s algorithm, not just trust in it (Prompt C:
    "no train/test leakage")."""
    split_of_player: dict[str, int] = {}
    for split_idx, split in enumerate(splits):
        for sample in split:
            existing = split_of_player.get(sample.player_id)
            if existing is not None and existing != split_idx:
                raise ValueError(
                    f"player_id={sample.player_id!r} appears in multiple splits "
                    f"(indices {existing} and {split_idx}) -- train/test leakage"
                )
            split_of_player[sample.player_id] = split_idx


def _build_windows(
    samples: Sequence[TrainingSample],
    feature_names: tuple[str, ...],
    normalization: NormalizationStats,
    config: TrainingConfig,
) -> list[Window]:
    windows: list[Window] = []
    for sample in samples:
        frame_ms = frame_timestamps_ms(sample.features)
        matrix = normalization.apply(feature_matrix(sample.features, feature_names))
        class_targets, start_targets, end_targets, outcome_targets = build_frame_targets(
            frame_ms, sample.trick_events
        )
        windows.extend(
            slice_windows(
                matrix,
                frame_ms,
                class_targets,
                start_targets,
                end_targets,
                outcome_targets,
                config.window_ms,
                config.stride_ms,
            )
        )
    return windows


def compute_class_pos_weight(windows: Sequence[Window], config: TrainingConfig) -> np.ndarray:
    """Per-class BCE `pos_weight` for the multi-label classification head
    (Prompt C: "class-balanced sampling or loss weighting"), fit on
    `windows` only -- callers must pass *training*-split windows so
    validation/test never leak into the weighting."""
    if config.class_balance_strategy == "none" or not windows:
        return np.ones(NUM_CLASSES, dtype=np.float32)
    stacked = np.concatenate([window.class_targets for window in windows], axis=0)
    positive = stacked.sum(axis=0)
    negative = stacked.shape[0] - positive
    pos_weight = np.where(positive > 0, negative / np.maximum(positive, 1.0), 1.0)
    return np.clip(pos_weight, 1.0, config.max_pos_weight).astype(np.float32)


def set_deterministic_seed(torch_module: Any, seed: int) -> None:
    """Seeds every RNG this package's training loop touches (Prompt C:
    "deterministic seed"). `use_deterministic_algorithms(warn_only=True)`
    is best-effort -- some ops (e.g. certain GroupNorm backward paths) have
    no deterministic CPU kernel; warning rather than raising keeps training
    runnable everywhere while still requesting determinism where available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(True, warn_only=True)


def _window_loss(
    torch_module: Any,
    model: Any,
    window: Window,
    pos_weight: Any,
    boundary_pos_weight: Any,
) -> Any:
    """Unweighted sum of the three head losses for one window -- a simple,
    documented default (not a tuned multi-task loss balance) appropriate to
    Prompt C's "modest and reproducible baseline"."""
    nn = torch_module.nn
    features = torch_module.from_numpy(window.features).float().unsqueeze(0)
    outputs = model(features)

    class_targets = torch_module.from_numpy(window.class_targets).float().unsqueeze(0)
    start_targets = torch_module.from_numpy(window.start_targets).float().unsqueeze(0)
    end_targets = torch_module.from_numpy(window.end_targets).float().unsqueeze(0)
    outcome_targets = torch_module.from_numpy(window.outcome_targets).long().unsqueeze(0)

    class_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["class_logits"], class_targets, pos_weight=pos_weight
    )
    start_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["start_logits"], start_targets, pos_weight=boundary_pos_weight
    )
    end_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["end_logits"], end_targets, pos_weight=boundary_pos_weight
    )

    outcome_logits = outputs["outcome_logits"].reshape(-1, NUM_OUTCOMES)
    outcome_flat = outcome_targets.reshape(-1)
    if bool((outcome_flat != -1).any()):
        outcome_loss = nn.functional.cross_entropy(outcome_logits, outcome_flat, ignore_index=-1)
    else:
        # Every frame in this window falls outside any event span -- no
        # outcome target to learn from; contribute 0 rather than the NaN
        # `cross_entropy` would produce from averaging over zero elements.
        outcome_loss = torch_module.zeros((), dtype=outcome_logits.dtype)

    return class_loss + start_loss + end_loss + outcome_loss


def _run_epoch(
    torch_module: Any,
    model: Any,
    windows: Sequence[Window],
    pos_weight: Any,
    boundary_pos_weight: Any,
    config: TrainingConfig,
    optimizer: Any | None,
    epoch_seed: int,
) -> float:
    """Runs one pass over `windows` -- training (with backprop + optimizer
    steps) if `optimizer` is given, evaluation otherwise -- and returns the
    mean per-window total loss."""
    is_training = optimizer is not None
    model.train(is_training)

    order = list(range(len(windows)))
    if optimizer is not None:
        random.Random(epoch_seed).shuffle(order)
        optimizer.zero_grad()

    total_loss = 0.0
    accumulated = 0
    for step, idx in enumerate(order):
        window = windows[idx]
        with torch_module.set_grad_enabled(is_training):
            loss = _window_loss(torch_module, model, window, pos_weight, boundary_pos_weight)
        total_loss += float(loss.detach())

        if optimizer is not None:
            (loss / config.batch_size).backward()
            accumulated += 1
            if accumulated >= config.batch_size or step == len(order) - 1:
                optimizer.step()
                optimizer.zero_grad()
                accumulated = 0

    return total_loss / len(windows) if windows else 0.0


def _collect_class_logits(
    torch_module: Any, model: Any, windows: Sequence[Window]
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenates every window's `class_logits`/`class_targets` -- the
    population `calibration.fit_temperature_per_class` fits on."""
    logits_chunks: list[np.ndarray] = []
    targets_chunks: list[np.ndarray] = []
    with torch_module.no_grad():
        for window in windows:
            features = torch_module.from_numpy(window.features).float().unsqueeze(0)
            output = model(features)
            logits_chunks.append(output["class_logits"].squeeze(0).cpu().numpy())
            targets_chunks.append(window.class_targets)
    return np.concatenate(logits_chunks, axis=0), np.concatenate(targets_chunks, axis=0)


class _ModelDetector:
    """Adapts a trained model + its inference-time context into the
    `interfaces.TemporalEventDetector.predict` shape (`FeatureSet ->
    (predictions, deductions)`), so `metrics.evaluate_detector` -- built for
    any such detector, not just this one model -- can score this in-training
    model exactly like it would score `detector_torch.PyTorchTemporalEventDetector`
    or a `baselines.py` reference detector. Always returns an empty
    deduction list, same rationale as `convert.to_analysis_event_prediction`
    and `detector_torch.py`: Prompt C's classes exclude equipment events."""

    def __init__(
        self,
        torch_module: Any,
        model: Any,
        feature_names: tuple[str, ...],
        normalization: NormalizationStats,
        calibration_temperatures: np.ndarray,
        model_name: str,
        model_version: str,
        inference_config: InferenceConfig,
    ) -> None:
        self._torch_module = torch_module
        self._model = model
        self._feature_names = feature_names
        self._normalization = normalization
        self._calibration_temperatures = calibration_temperatures
        self.model_name = model_name
        self.model_version = model_version
        self._inference_config = inference_config

    def predict(
        self, features: FeatureSet
    ) -> tuple[list[AnalysisEventPrediction], list[DeductionPrediction]]:
        predictions = run_inference(
            self._torch_module,
            self._model,
            features,
            self._feature_names,
            self._normalization,
            self._calibration_temperatures,
            self.model_name,
            self.model_version,
            self._inference_config,
        )
        return predictions, []


@dataclass(slots=True, frozen=True)
class TrainingRunResult:
    """Everything one `train_model` call produces: the trained model itself
    (best-epoch weights already loaded) plus every artefact
    `checkpoint.save_checkpoint`/`EventModelMetadata` needs."""

    model: Any
    torch_module: Any
    config: TrainingConfig
    inference_config: InferenceConfig
    feature_names: tuple[str, ...]
    normalization: NormalizationStats
    calibration_temperatures: np.ndarray
    model_name: str
    model_version: str
    player_splits: dict[str, str]
    best_epoch: int
    val_loss_history: list[float]
    val_report: EvaluationReport
    test_report: EvaluationReport | None
    train_sample_count: int
    val_sample_count: int
    test_sample_count: int


def train_model(
    samples: Sequence[TrainingSample],
    config: TrainingConfig | None = None,
    inference_config: InferenceConfig | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> TrainingRunResult:
    """Trains one `TrickEventTCN` end to end: player-grouped split with a
    leakage check, per-column normalization fit on the training split only,
    class-balanced classification loss, early stopping on validation loss,
    post-hoc per-class temperature calibration on the validation split, and
    a full `metrics.evaluate()` report on both validation and (if non-empty)
    test splits. Returns everything needed to checkpoint and later serve
    the result -- this function itself never touches the filesystem."""
    config = config or TrainingConfig()
    inference_config = inference_config or InferenceConfig()
    torch_module = import_torch()
    set_deterministic_seed(torch_module, config.seed)

    train_samples, val_samples, test_samples = player_grouped_split(
        samples, config.seed, config.train_ratio, config.val_ratio
    )
    assert_no_leakage(train_samples, val_samples, test_samples)
    if not train_samples or not val_samples:
        distinct_players = len({sample.player_id for sample in samples})
        raise ValueError(
            f"Not enough distinct players ({distinct_players}) to form non-empty "
            "train and val splits at the configured train_ratio/val_ratio."
        )

    feature_names = FEATURE_SUBSETS[config.feature_subset]
    normalization = fit_normalization(
        [feature_matrix(sample.features, feature_names) for sample in train_samples]
    )

    train_windows = _build_windows(train_samples, feature_names, normalization, config)
    val_windows = _build_windows(val_samples, feature_names, normalization, config)
    if not train_windows or not val_windows:
        raise ValueError("Not enough frames across train/val samples to build any window.")

    pos_weight = torch_module.from_numpy(compute_class_pos_weight(train_windows, config)).float()
    boundary_pos_weight = torch_module.full(
        (NUM_CLASSES,), _BOUNDARY_POS_WEIGHT, dtype=torch_module.float32
    )

    model = build_model(torch_module, len(feature_names), config)
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
            train_windows,
            pos_weight,
            boundary_pos_weight,
            config,
            optimizer=optimizer,
            epoch_seed=config.seed + epoch,
        )
        val_loss = _run_epoch(
            torch_module,
            model,
            val_windows,
            pos_weight,
            boundary_pos_weight,
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

    val_class_logits, val_class_targets = _collect_class_logits(torch_module, model, val_windows)
    calibration_temperatures = fit_temperature_per_class(val_class_logits, val_class_targets)

    model_version = f"{config.feature_subset}-tcn-seed{config.seed}-epoch{best_epoch}"
    detector = _ModelDetector(
        torch_module,
        model,
        feature_names,
        normalization,
        calibration_temperatures,
        model_name,
        model_version,
        inference_config,
    )

    val_report = evaluate_detector(
        detector,
        val_samples,
        tiou_thresholds_map=inference_config.tiou_thresholds,
        calibration_bins=inference_config.calibration_bins,
    )
    test_report: EvaluationReport | None = (
        evaluate_detector(
            detector,
            test_samples,
            tiou_thresholds_map=inference_config.tiou_thresholds,
            calibration_bins=inference_config.calibration_bins,
        )
        if test_samples
        else None
    )

    player_splits = {sample.player_id: "train" for sample in train_samples}
    player_splits.update({sample.player_id: "val" for sample in val_samples})
    player_splits.update({sample.player_id: "test" for sample in test_samples})

    return TrainingRunResult(
        model=model,
        torch_module=torch_module,
        config=config,
        inference_config=inference_config,
        feature_names=feature_names,
        normalization=normalization,
        calibration_temperatures=calibration_temperatures,
        model_name=model_name,
        model_version=model_version,
        player_splits=player_splits,
        best_epoch=best_epoch,
        val_loss_history=val_loss_history,
        val_report=val_report,
        test_report=test_report,
        train_sample_count=len(train_samples),
        val_sample_count=len(val_samples),
        test_sample_count=len(test_samples),
    )
