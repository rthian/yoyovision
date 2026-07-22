"""Tests for `yoyovision_ml.multimodal.features` -- Prompt E's feature-column
constants and `fuse_feature_sets` merge step.

Named `test_multimodal_features.py` rather than `test_features.py` to avoid a
pytest module-basename collision with `tests/perception/test_features.py`
(neither `tests/` package has `__init__.py` files, so pytest's rootdir-
relative import needs every test module's basename to be unique)."""

from __future__ import annotations

from yoyovision_ml.domain import FeatureFrame, FeatureSet
from yoyovision_ml.multimodal.features import (
    AUDIO_FEATURE_NAMES,
    MULTIMODAL_FEATURE_NAMES,
    RGB_FEATURE_NAMES,
    STRING_SEGMENTATION_FEATURE_NAMES,
    fuse_feature_sets,
)


def _feature_set(
    frame_ms_to_values: dict[int, dict[str, float]], feature_names: tuple[str, ...]
) -> FeatureSet:
    frames = tuple(
        FeatureFrame(frame_ms=ms, values=values) for ms, values in frame_ms_to_values.items()
    )
    return FeatureSet(frames=frames, feature_names=feature_names, fps=10.0)


def test_multimodal_feature_names_concatenates_all_three_modalities_in_order() -> None:
    assert MULTIMODAL_FEATURE_NAMES == (
        RGB_FEATURE_NAMES + STRING_SEGMENTATION_FEATURE_NAMES + AUDIO_FEATURE_NAMES
    )


def test_multimodal_feature_names_has_no_duplicates() -> None:
    assert len(MULTIMODAL_FEATURE_NAMES) == len(set(MULTIMODAL_FEATURE_NAMES))


class TestFuseFeatureSets:
    def test_fuse_merges_exact_matching_timestamps(self) -> None:
        kinematic = _feature_set(
            {0: {"yoyo_speed": 1.0}, 100: {"yoyo_speed": 2.0}}, ("yoyo_speed",)
        )
        rgb = _feature_set({0: {"rgb_embed_0": 0.5}, 100: {"rgb_embed_0": 0.6}}, ("rgb_embed_0",))
        string_seg = _feature_set({0: {"string_seg_angle_deg": 10.0}}, ("string_seg_angle_deg",))
        audio = _feature_set({0: {"audio_tempo_bpm": 120.0}}, ("audio_tempo_bpm",))

        fused = fuse_feature_sets(kinematic, rgb, string_seg, audio)

        assert [f.frame_ms for f in fused.frames] == [0, 100]
        frame_0 = fused.frames[0]
        assert frame_0.values["yoyo_speed"] == 1.0
        assert frame_0.values["rgb_embed_0"] == 0.5
        assert frame_0.values["string_seg_angle_deg"] == 10.0
        assert frame_0.values["audio_tempo_bpm"] == 120.0
        # Frame at 100ms has no string_seg/audio sample, but nearest-neighbor
        # lookup still pulls in the frame_ms=0 values since they're the only
        # candidates.
        frame_100 = fused.frames[1]
        assert frame_100.values["yoyo_speed"] == 2.0
        assert frame_100.values["rgb_embed_0"] == 0.6

    def test_fuse_uses_nearest_frame_when_timestamps_do_not_align(self) -> None:
        kinematic = _feature_set({50: {"yoyo_speed": 1.0}}, ("yoyo_speed",))
        rgb = _feature_set(
            {0: {"rgb_embed_0": 0.1}, 1000: {"rgb_embed_0": 0.9}}, ("rgb_embed_0",)
        )
        empty_string_seg = _feature_set({}, ("string_seg_angle_deg",))
        empty_audio = _feature_set({}, ("audio_tempo_bpm",))

        fused = fuse_feature_sets(kinematic, rgb, empty_string_seg, empty_audio)

        assert len(fused.frames) == 1
        # 50ms is nearer to the rgb sample at 0ms than the one at 1000ms.
        assert fused.frames[0].values["rgb_embed_0"] == 0.1

    def test_fuse_with_no_multimodal_frames_leaves_kinematic_features_unchanged(self) -> None:
        kinematic = _feature_set({0: {"yoyo_speed": 1.0}}, ("yoyo_speed",))
        empty = _feature_set({}, ("rgb_embed_0",))

        fused = fuse_feature_sets(kinematic, empty, empty, empty)

        assert len(fused.frames) == 1
        assert fused.frames[0].values == {"yoyo_speed": 1.0}

    def test_fuse_feature_names_includes_kinematic_and_observed_multimodal_columns(self) -> None:
        kinematic = _feature_set({0: {"yoyo_speed": 1.0}}, ("yoyo_speed",))
        rgb = _feature_set({0: {"rgb_embed_0": 0.5}}, ("rgb_embed_0",))
        string_seg = _feature_set({0: {"string_seg_angle_deg": 10.0}}, ("string_seg_angle_deg",))
        audio = _feature_set({0: {"audio_tempo_bpm": 120.0}}, ("audio_tempo_bpm",))

        fused = fuse_feature_sets(kinematic, rgb, string_seg, audio)

        assert set(fused.feature_names) == {
            "yoyo_speed",
            "rgb_embed_0",
            "string_seg_angle_deg",
            "audio_tempo_bpm",
        }

    def test_fuse_preserves_kinematic_fps(self) -> None:
        kinematic = FeatureSet(
            frames=(FeatureFrame(frame_ms=0, values={"yoyo_speed": 1.0}),),
            feature_names=("yoyo_speed",),
            fps=15.0,
        )
        empty = _feature_set({}, ())

        fused = fuse_feature_sets(kinematic, empty, empty, empty)

        assert fused.fps == 15.0

    def test_fuse_with_no_kinematic_frames_produces_no_output_frames(self) -> None:
        kinematic = _feature_set({}, ("yoyo_speed",))
        rgb = _feature_set({0: {"rgb_embed_0": 0.5}}, ("rgb_embed_0",))
        empty = _feature_set({}, ())

        fused = fuse_feature_sets(kinematic, rgb, empty, empty)

        assert fused.frames == ()
