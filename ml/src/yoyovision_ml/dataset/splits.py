"""Player-grouped train/val/test split generation with a fixed random seed.

A player's videos must never straddle two splits -- otherwise a model can
memorize a specific performer's body proportions/style/background rather
than generalizing, and evaluation numbers become optimistic. Splitting is
therefore done by *player*, not by *video* or *event*.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from yoyovision_ml.dataset.schema import DatasetVideo, SplitName

DEFAULT_RATIOS: dict[SplitName, float] = {
    SplitName.TRAIN: 0.7,
    SplitName.VAL: 0.15,
    SplitName.TEST: 0.15,
}


@dataclass(slots=True, frozen=True)
class SplitAssignment:
    #: video_id -> split
    video_splits: dict[str, SplitName]
    #: player_id -> split (every one of a player's videos always agrees with this)
    player_splits: dict[str, SplitName]
    seed: int
    ratios: dict[SplitName, float]


def _validate_ratios(ratios: dict[SplitName, float]) -> None:
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total} ({ratios}).")
    if any(r < 0 for r in ratios.values()):
        raise ValueError(f"Split ratios must be non-negative: {ratios}")


def generate_player_grouped_splits(
    videos: list[DatasetVideo],
    *,
    seed: int = 42,
    ratios: dict[SplitName, float] | None = None,
) -> SplitAssignment:
    """Deterministically assigns every player (and therefore every one of
    their videos) to exactly one split.

    Deterministic for a given `(videos, seed, ratios)` triple: players are
    sorted by ID before shuffling with a seeded `random.Random`, so the
    result never depends on dict/set iteration order or on the order videos
    were passed in.
    """
    ratios = ratios or DEFAULT_RATIOS
    _validate_ratios(ratios)

    videos_by_player: dict[str, list[DatasetVideo]] = defaultdict(list)
    for video in videos:
        videos_by_player[video.player_id].append(video)

    player_ids = sorted(videos_by_player)
    rng = random.Random(seed)
    rng.shuffle(player_ids)

    total_video_count = len(videos)
    target_counts = {split: round(ratio * total_video_count) for split, ratio in ratios.items()}

    player_splits: dict[str, SplitName] = {}
    running_counts: dict[SplitName, int] = dict.fromkeys(ratios, 0)
    split_order = sorted(ratios, key=lambda s: -ratios[s])  # fill largest-quota split first

    for player_id in player_ids:
        player_video_count = len(videos_by_player[player_id])
        # Assign to whichever split is furthest below its target (as a
        # fraction of target), tie-broken by split_order, so no single
        # split systematically starves as players are consumed.
        best_split = min(
            split_order,
            key=lambda s: (
                (running_counts[s] / target_counts[s]) if target_counts[s] > 0 else float("inf"),
            ),
        )
        player_splits[player_id] = best_split
        running_counts[best_split] += player_video_count

    video_splits = {video.video_id: player_splits[video.player_id] for video in videos}

    return SplitAssignment(
        video_splits=video_splits,
        player_splits=player_splits,
        seed=seed,
        ratios=dict(ratios),
    )


def find_leaked_players(
    videos: list[DatasetVideo], video_splits: dict[str, SplitName]
) -> dict[str, set[SplitName]]:
    """Returns `{player_id: {splits they appear in}}` for every player
    appearing in more than one split -- empty dict means no leakage."""
    player_splits: dict[str, set[SplitName]] = defaultdict(set)
    for video in videos:
        split = video_splits.get(video.video_id)
        if split is not None:
            player_splits[video.player_id].add(split)
    return {player: splits for player, splits in player_splits.items() if len(splits) > 1}
