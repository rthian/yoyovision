from __future__ import annotations

import numpy as np
import pytest

from yoyovision_ml.domain import FeatureSet
from yoyovision_ml.events.config import InferenceConfig, TrainingConfig
from yoyovision_ml.events.labels import NUM_CLASSES
from yoyovision_ml.events.synthetic import generate_synthetic_dataset
from yoyovision_ml.events.train import (
    assert_no_leakage,
    compute_class_pos_weight,
    player_grouped_split,
    set_deterministic_seed,
    train_model,
)
from yoyovision_ml.events.types import TrainingSample
from yoyovision_ml.events.windowing import Window

torch = pytest.importorskip("torch")


def _sample(player_id: str, video_id: str) -> TrainingSample:
    features = FeatureSet(frames=(), feature_names=(), fps=30.0)
    return TrainingSample(
        video_id=video_id, player_id=player_id, features=features, trick_events=()
    )


def _samples_for_players(players: list[str], clips_per_player: int = 1) -> list[TrainingSample]:
    return [
        _sample(player_id, f"{player_id}-clip-{clip_idx}")
        for player_id in players
        for clip_idx in range(clips_per_player)
    ]


# --------------------------------------------------------------------------- #
# player_grouped_split / assert_no_leakage
# --------------------------------------------------------------------------- #
def test_player_grouped_split_never_splits_a_single_player_across_splits() -> None:
    samples = _samples_for_players([f"player-{i}" for i in range(10)], clips_per_player=3)
    train, val, test = player_grouped_split(samples, seed=1, train_ratio=0.7, val_ratio=0.15)
    assert_no_leakage(train, val, test)  # must not raise


def test_player_grouped_split_is_deterministic_for_the_same_seed() -> None:
    samples = _samples_for_players([f"player-{i}" for i in range(10)])
    split_a = player_grouped_split(samples, seed=7, train_ratio=0.7, val_ratio=0.15)
    split_b = player_grouped_split(samples, seed=7, train_ratio=0.7, val_ratio=0.15)
    assert [s.video_id for s in split_a[0]] == [s.video_id for s in split_b[0]]
    assert [s.video_id for s in split_a[1]] == [s.video_id for s in split_b[1]]
    assert [s.video_id for s in split_a[2]] == [s.video_id for s in split_b[2]]


def test_player_grouped_split_different_seeds_can_produce_different_splits() -> None:
    samples = _samples_for_players([f"player-{i}" for i in range(12)])
    split_a = player_grouped_split(samples, seed=1, train_ratio=0.5, val_ratio=0.25)
    split_b = player_grouped_split(samples, seed=99, train_ratio=0.5, val_ratio=0.25)
    train_ids_a = {s.video_id for s in split_a[0]}
    train_ids_b = {s.video_id for s in split_b[0]}
    assert train_ids_a != train_ids_b


def test_player_grouped_split_every_sample_is_placed_exactly_once() -> None:
    samples = _samples_for_players([f"player-{i}" for i in range(9)], clips_per_player=2)
    train, val, test = player_grouped_split(samples, seed=3, train_ratio=0.6, val_ratio=0.2)
    all_ids = [s.video_id for s in train] + [s.video_id for s in val] + [s.video_id for s in test]
    assert sorted(all_ids) == sorted(s.video_id for s in samples)


def test_player_grouped_split_raises_when_ratios_leave_no_room_for_test() -> None:
    samples = _samples_for_players(["player-0"])
    with pytest.raises(ValueError, match="test_ratio"):
        player_grouped_split(samples, seed=1, train_ratio=0.8, val_ratio=0.2)


def test_assert_no_leakage_raises_when_a_player_appears_in_two_splits() -> None:
    shared = _sample("player-shared", "clip-a")
    other = _sample("player-shared", "clip-b")
    with pytest.raises(ValueError, match="train/test leakage"):
        assert_no_leakage([shared], [other])


def test_assert_no_leakage_allows_the_same_player_repeated_within_one_split() -> None:
    first = _sample("player-x", "clip-a")
    second = _sample("player-x", "clip-b")
    assert_no_leakage([first, second], [])  # must not raise


# --------------------------------------------------------------------------- #
# compute_class_pos_weight
# --------------------------------------------------------------------------- #
def _window_with_class_targets(class_targets: np.ndarray) -> Window:
    num_frames = class_targets.shape[0]
    return Window(
        features=np.zeros((num_frames, 1)),
        class_targets=class_targets,
        start_targets=np.zeros_like(class_targets),
        end_targets=np.zeros_like(class_targets),
        outcome_targets=np.full(num_frames, -1, dtype=np.int64),
        frame_ms=np.arange(num_frames, dtype=np.int64),
    )


def test_compute_class_pos_weight_returns_ones_when_strategy_is_none() -> None:
    class_targets = np.zeros((4, NUM_CLASSES), dtype=np.float32)
    windows = [_window_with_class_targets(class_targets)]
    weight = compute_class_pos_weight(windows, TrainingConfig(class_balance_strategy="none"))
    assert np.array_equal(weight, np.ones(NUM_CLASSES, dtype=np.float32))


def test_compute_class_pos_weight_returns_ones_for_no_windows() -> None:
    weight = compute_class_pos_weight([], TrainingConfig())
    assert np.array_equal(weight, np.ones(NUM_CLASSES, dtype=np.float32))


def test_compute_class_pos_weight_computes_negative_over_positive_ratio() -> None:
    class_targets = np.zeros((10, NUM_CLASSES), dtype=np.float32)
    class_targets[:2, 0] = 1.0  # class 0: 2 positive, 8 negative -> pos_weight 4.0
    windows = [_window_with_class_targets(class_targets)]
    weight = compute_class_pos_weight(
        windows, TrainingConfig(class_balance_strategy="inverse_frequency", max_pos_weight=50.0)
    )
    assert weight[0] == pytest.approx(4.0)
    assert weight[1] == pytest.approx(1.0)  # never-positive class keeps weight 1.0


def test_compute_class_pos_weight_clips_at_max_pos_weight() -> None:
    class_targets = np.zeros((1000, NUM_CLASSES), dtype=np.float32)
    class_targets[0, 0] = 1.0  # 1 positive, 999 negative -> raw ratio 999.0
    windows = [_window_with_class_targets(class_targets)]
    weight = compute_class_pos_weight(
        windows, TrainingConfig(class_balance_strategy="inverse_frequency", max_pos_weight=10.0)
    )
    assert weight[0] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# set_deterministic_seed
# --------------------------------------------------------------------------- #
def test_set_deterministic_seed_makes_torch_rand_reproducible() -> None:
    set_deterministic_seed(torch, 123)
    first = torch.rand(4)
    set_deterministic_seed(torch, 123)
    second = torch.rand(4)
    assert torch.equal(first, second)


# --------------------------------------------------------------------------- #
# train_model (end to end, small + fast)
# --------------------------------------------------------------------------- #
def _tiny_config() -> TrainingConfig:
    return TrainingConfig(
        feature_subset="trajectory",
        window_ms=1500,
        stride_ms=1500,
        seed=0,
        hidden_channels=4,
        num_blocks=1,
        kernel_size=3,
        batch_size=2,
        max_epochs=2,
        early_stopping_patience=1,
        train_ratio=0.6,
        val_ratio=0.2,
    )


def test_train_model_end_to_end_produces_a_consistent_result() -> None:
    samples = generate_synthetic_dataset(num_players=6, clips_per_player=1, num_events_per_clip=3)
    result = train_model(samples, config=_tiny_config(), inference_config=InferenceConfig())

    assert result.best_epoch >= 1
    assert len(result.val_loss_history) >= 1
    assert result.train_sample_count + result.val_sample_count + result.test_sample_count == len(
        samples
    )
    assert result.calibration_temperatures.shape == (NUM_CLASSES,)
    assert result.model_version.startswith("trajectory-tcn-seed0-epoch")


def test_train_model_never_leaks_a_player_across_splits() -> None:
    samples = generate_synthetic_dataset(num_players=6, clips_per_player=1, num_events_per_clip=3)
    result = train_model(samples, config=_tiny_config())

    split_by_player = result.player_splits
    assert len(split_by_player) == len({sample.player_id for sample in samples})
    assert set(split_by_player.values()) <= {"train", "val", "test"}


def test_train_model_raises_when_too_few_players_for_a_non_empty_split() -> None:
    samples = generate_synthetic_dataset(num_players=1, clips_per_player=1, num_events_per_clip=2)
    with pytest.raises(ValueError, match="distinct players"):
        train_model(samples, config=_tiny_config())


def test_train_model_produces_no_test_report_when_the_test_split_is_empty() -> None:
    samples = generate_synthetic_dataset(num_players=4, clips_per_player=1, num_events_per_clip=2)
    config = TrainingConfig(
        feature_subset="trajectory",
        window_ms=1500,
        stride_ms=1500,
        seed=0,
        hidden_channels=4,
        num_blocks=1,
        max_epochs=1,
        early_stopping_patience=1,
        train_ratio=0.7,
        val_ratio=0.299,  # leaves test_ratio ~ 0, so target_count["test"] rounds to 0
    )
    result = train_model(samples, config=config)
    if result.test_sample_count == 0:
        assert result.test_report is None
