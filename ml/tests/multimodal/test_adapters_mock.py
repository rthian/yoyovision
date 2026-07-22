"""Tests for Prompt E's deterministic mock adapters
(`yoyovision_ml.multimodal.adapters_mock`)."""

from __future__ import annotations

from pathlib import Path

from yoyovision_ml.domain import BoundingBox, Track
from yoyovision_ml.interfaces import FrameRef
from yoyovision_ml.multimodal.adapters_mock import (
    MockAudioAnalyzer,
    MockRgbEncoder,
    MockStringSegmenter,
)
from yoyovision_ml.multimodal.features import (
    AUDIO_FEATURE_NAMES,
    RGB_FEATURE_NAMES,
    STRING_SEGMENTATION_FEATURE_NAMES,
)


def _frames(frame_ms_values: list[int]) -> list[FrameRef]:
    return [FrameRef(frame_ms=ms, array=None) for ms in frame_ms_values]


def _track(frame_ms: int, x: float = 10.0) -> Track:
    return Track(
        track_id="t0",
        frame_ms=frame_ms,
        bbox=BoundingBox(x=x, y=5.0, width=20.0, height=20.0),
        confidence=0.9,
        class_label="yoyo",
    )


class TestMockRgbEncoder:
    def test_encode_is_deterministic_given_same_frames(self) -> None:
        encoder = MockRgbEncoder()
        frames = _frames([0, 100, 200])

        result_a = encoder.encode(frames)
        result_b = encoder.encode(frames)

        assert result_a == result_b

    def test_encode_produces_one_feature_frame_per_input_frame(self) -> None:
        encoder = MockRgbEncoder()
        frames = _frames([0, 100, 200])

        result = encoder.encode(frames)

        assert len(result.frames) == 3
        assert [f.frame_ms for f in result.frames] == [0, 100, 200]
        assert result.feature_names == RGB_FEATURE_NAMES
        for frame in result.frames:
            assert set(frame.values) == set(RGB_FEATURE_NAMES)

    def test_encode_ignores_frame_array_contents(self) -> None:
        """Real pixel content must not affect output -- these are mocks."""
        encoder = MockRgbEncoder()
        frames_a = [FrameRef(frame_ms=0, array="anything")]
        frames_b = [FrameRef(frame_ms=0, array=[1, 2, 3])]

        assert encoder.encode(frames_a) == encoder.encode(frames_b)

    def test_encode_handles_empty_batch(self) -> None:
        encoder = MockRgbEncoder()
        result = encoder.encode([])
        assert result.frames == ()
        assert result.fps == 0.0

    def test_model_name_and_version_are_mock_labelled(self) -> None:
        encoder = MockRgbEncoder()
        assert encoder.model_name.startswith("mock-")
        assert encoder.model_version.endswith("-mock")


class TestMockStringSegmenter:
    def test_segment_is_deterministic_given_same_inputs(self) -> None:
        segmenter = MockStringSegmenter()
        frames = _frames([0, 100])
        tracks = [_track(0), _track(100)]

        result_a = segmenter.segment(frames, tracks)
        result_b = segmenter.segment(frames, tracks)

        assert result_a == result_b

    def test_segment_only_emits_frames_with_a_matching_track(self) -> None:
        segmenter = MockStringSegmenter()
        frames = _frames([0, 100, 200])
        tracks = [_track(0), _track(200)]  # no track at frame_ms=100

        result = segmenter.segment(frames, tracks)

        assert [f.frame_ms for f in result.frames] == [0, 200]
        assert result.feature_names == STRING_SEGMENTATION_FEATURE_NAMES

    def test_segment_output_is_sensitive_to_track_position(self) -> None:
        segmenter = MockStringSegmenter()
        frames = _frames([0])

        result_near_origin = segmenter.segment(frames, [_track(0, x=10.0)])
        result_far = segmenter.segment(frames, [_track(0, x=900.0)])

        assert result_near_origin.frames[0].values != result_far.frames[0].values

    def test_segment_handles_no_tracks(self) -> None:
        segmenter = MockStringSegmenter()
        result = segmenter.segment(_frames([0, 100]), [])
        assert result.frames == ()


class TestMockAudioAnalyzer:
    def test_analyze_is_deterministic_given_same_inputs(self) -> None:
        analyzer = MockAudioAnalyzer()
        video_path = Path("/tmp/does-not-need-to-exist.mp4")

        result_a = analyzer.analyze(video_path, duration_ms=5_000)
        result_b = analyzer.analyze(video_path, duration_ms=5_000)

        assert result_a == result_b

    def test_analyze_covers_the_full_duration_at_fixed_rate(self) -> None:
        analyzer = MockAudioAnalyzer()
        result = analyzer.analyze(Path("/tmp/clip.mp4"), duration_ms=5_000)

        assert result.fps > 0
        assert len(result.frames) > 0
        assert result.frames[-1].frame_ms < 5_000
        assert result.feature_names == AUDIO_FEATURE_NAMES

    def test_analyze_handles_zero_duration(self) -> None:
        analyzer = MockAudioAnalyzer()
        result = analyzer.analyze(Path("/tmp/clip.mp4"), duration_ms=0)
        assert result.frames == ()
        assert result.fps == 0.0

    def test_analyze_differs_by_video_path(self) -> None:
        analyzer = MockAudioAnalyzer()
        result_a = analyzer.analyze(Path("/tmp/a.mp4"), duration_ms=1_000)
        result_b = analyzer.analyze(Path("/tmp/b.mp4"), duration_ms=1_000)
        assert result_a.frames != result_b.frames
