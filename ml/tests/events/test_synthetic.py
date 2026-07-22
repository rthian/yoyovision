from __future__ import annotations

from itertools import pairwise

from yoyovision_ml.events.labels import EVENT_CLASSES, FEATURE_SUBSETS, OUTCOME_CLASSES
from yoyovision_ml.events.synthetic import generate_synthetic_dataset, generate_synthetic_sample
from yoyovision_ml.multimodal.features import MULTIMODAL_FEATURE_NAMES
from yoyovision_ml.perception.features import ALL_FEATURE_NAMES


def test_generate_synthetic_sample_is_deterministic_for_same_arguments() -> None:
    first = generate_synthetic_sample(seed=1, video_id="video-a", player_id="player-a")
    second = generate_synthetic_sample(seed=1, video_id="video-a", player_id="player-a")

    assert first.trick_events == second.trick_events
    assert first.features.frames == second.features.frames


def test_generate_synthetic_sample_differs_with_a_different_seed() -> None:
    first = generate_synthetic_sample(seed=1, video_id="video-a", player_id="player-a")
    second = generate_synthetic_sample(seed=2, video_id="video-a", player_id="player-a")

    assert first.trick_events[0].family != second.trick_events[0].family or (
        first.features.frames != second.features.frames
    )


def test_generate_synthetic_sample_events_are_non_overlapping_and_ordered() -> None:
    sample = generate_synthetic_sample(seed=5, video_id="video-b", player_id="player-b")
    events = sample.trick_events
    assert len(events) > 1
    for earlier, later in pairwise(events):
        assert earlier.end_ms <= later.start_ms


def test_generate_synthetic_sample_cycles_through_every_class_and_outcome() -> None:
    sample = generate_synthetic_sample(
        seed=0, video_id="video-c", player_id="player-c", num_events=len(EVENT_CLASSES)
    )
    families = {event.family for event in sample.trick_events}
    outcomes = {str(event.outcome) for event in sample.trick_events}
    assert families == set(EVENT_CLASSES)
    assert outcomes == set(OUTCOME_CLASSES)


def test_generate_synthetic_sample_feature_set_covers_full_clip_duration() -> None:
    sample = generate_synthetic_sample(seed=3, video_id="video-d", player_id="player-d", fps=30.0)
    assert len(sample.features.frames) > 0
    last_event_end = max(event.end_ms for event in sample.trick_events)
    assert sample.features.frames[-1].frame_ms >= last_event_end


def test_generate_synthetic_sample_carries_video_and_player_id_through() -> None:
    sample = generate_synthetic_sample(seed=9, video_id="my-video", player_id="my-player")
    assert sample.video_id == "my-video"
    assert sample.player_id == "my-player"


def test_generate_synthetic_dataset_produces_expected_sample_count() -> None:
    samples = generate_synthetic_dataset(num_players=3, clips_per_player=2, seed=42)
    assert len(samples) == 6


def test_generate_synthetic_dataset_assigns_distinct_player_ids() -> None:
    samples = generate_synthetic_dataset(num_players=4, clips_per_player=1, seed=42)
    player_ids = {sample.player_id for sample in samples}
    assert len(player_ids) == 4


def test_generate_synthetic_dataset_is_deterministic_for_same_seed() -> None:
    first = generate_synthetic_dataset(num_players=2, clips_per_player=2, seed=7)
    second = generate_synthetic_dataset(num_players=2, clips_per_player=2, seed=7)
    assert [sample.video_id for sample in first] == [sample.video_id for sample in second]
    assert [sample.trick_events for sample in first] == [sample.trick_events for sample in second]


def test_generate_synthetic_sample_feature_set_includes_kinematic_and_multimodal_columns() -> None:
    """Prompt E: every synthetic `FeatureSet` carries both the kinematic
    (Prompt B) columns and the RGB/string-seg/audio (Prompt E) columns, so
    `FEATURE_SUBSETS["kinematics_only"]` and `[...]["multimodal_fused"]` can
    select sub-tuples of the *same* generated dataset."""
    sample = generate_synthetic_sample(seed=11, video_id="video-e", player_id="player-e")

    assert set(sample.features.feature_names) == set(ALL_FEATURE_NAMES) | set(
        MULTIMODAL_FEATURE_NAMES
    )
    for frame in sample.features.frames:
        assert set(frame.values) == set(sample.features.feature_names)


def test_generate_synthetic_sample_supports_selecting_kinematics_only_subset() -> None:
    sample = generate_synthetic_sample(seed=12, video_id="video-f", player_id="player-f")
    kinematics_only_names = FEATURE_SUBSETS["kinematics_only"]

    for frame in sample.features.frames:
        assert set(kinematics_only_names).issubset(frame.values)


def test_generate_synthetic_sample_supports_selecting_multimodal_fused_subset() -> None:
    sample = generate_synthetic_sample(seed=13, video_id="video-g", player_id="player-g")
    multimodal_fused_names = FEATURE_SUBSETS["multimodal_fused"]

    for frame in sample.features.frames:
        assert set(multimodal_fused_names).issubset(frame.values)


def test_generate_synthetic_sample_multimodal_columns_vary_by_class_signature() -> None:
    """The multimodal columns must not be flat noise disconnected from
    events -- same "class-signature envelope" treatment as the kinematic
    columns, so an ablation comparing `"kinematics_only"` vs.
    `"multimodal_fused"` is exercising real signal-bearing columns."""
    sample = generate_synthetic_sample(
        seed=14, video_id="video-h", player_id="player-h", num_events=len(EVENT_CLASSES)
    )
    one_multimodal_col = MULTIMODAL_FEATURE_NAMES[0]
    values = {frame.values[one_multimodal_col] for frame in sample.features.frames}
    assert len(values) > 1
