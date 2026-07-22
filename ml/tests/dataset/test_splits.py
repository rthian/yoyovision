from __future__ import annotations

from yoyovision_ml.dataset.schema import DatasetVideo, SplitName
from yoyovision_ml.dataset.splits import (
    DEFAULT_RATIOS,
    find_leaked_players,
    generate_player_grouped_splits,
)


def _videos(player_video_counts: dict[str, int]) -> list[DatasetVideo]:
    videos = []
    for player_id, count in player_video_counts.items():
        for i in range(count):
            videos.append(
                DatasetVideo(
                    video_id=f"{player_id}_v{i}",
                    player_id=player_id,
                    relative_path=f"videos/{player_id}_v{i}.mp4",
                    checksum_sha256="a" * 64,
                    duration_ms=10_000,
                    width=1920,
                    height=1080,
                    source_fps=30.0,
                )
            )
    return videos


def _many_players(n_players: int, videos_per_player: int = 1) -> list[DatasetVideo]:
    return _videos({f"player_{i:03d}": videos_per_player for i in range(n_players)})


def test_split_never_leaks_a_player_across_splits() -> None:
    videos = _videos({"p1": 3, "p2": 2, "p3": 1, "p4": 4, "p5": 1})
    assignment = generate_player_grouped_splits(videos, seed=42)

    leaks = find_leaked_players(videos, assignment.video_splits)
    assert leaks == {}

    for video in videos:
        assert assignment.video_splits[video.video_id] == assignment.player_splits[video.player_id]


def test_split_is_deterministic_for_same_seed() -> None:
    videos = _many_players(20, videos_per_player=2)
    first = generate_player_grouped_splits(videos, seed=7)
    second = generate_player_grouped_splits(videos, seed=7)
    assert first.video_splits == second.video_splits
    assert first.player_splits == second.player_splits


def test_split_can_differ_for_different_seeds() -> None:
    videos = _many_players(30, videos_per_player=1)
    first = generate_player_grouped_splits(videos, seed=1)
    second = generate_player_grouped_splits(videos, seed=2)
    assert first.player_splits != second.player_splits


def test_split_respects_ratios_approximately_with_enough_players() -> None:
    videos = _many_players(100, videos_per_player=1)
    assignment = generate_player_grouped_splits(videos, seed=42, ratios=DEFAULT_RATIOS)

    counts = {split: 0 for split in DEFAULT_RATIOS}
    for split in assignment.video_splits.values():
        counts[split] += 1

    total = len(videos)
    for split, ratio in DEFAULT_RATIOS.items():
        assert abs(counts[split] / total - ratio) < 0.05


def test_all_players_are_assigned_exactly_one_split() -> None:
    videos = _videos({"p1": 2, "p2": 2, "p3": 2})
    assignment = generate_player_grouped_splits(videos, seed=42)
    assert set(assignment.player_splits) == {"p1", "p2", "p3"}
    assert all(isinstance(s, SplitName) for s in assignment.player_splits.values())


def test_find_leaked_players_detects_injected_leak() -> None:
    videos = _videos({"p1": 2})
    # Deliberately construct a leak: same player's two videos in different splits.
    leaked_assignment = {videos[0].video_id: SplitName.TRAIN, videos[1].video_id: SplitName.VAL}
    leaks = find_leaked_players(videos, leaked_assignment)
    assert "p1" in leaks
    assert leaks["p1"] == {SplitName.TRAIN, SplitName.VAL}
