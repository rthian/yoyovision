"""Deterministic mock adapters for Prompt E (RGB, string, and audio fusion).

Same contract as the package-root `adapters_mock.py` (product principles
#6/#7): every class here is a stable hash-seeded function of its input, NOT
a trained model. `model_name` is `mock-`-prefixed and `model_version` is
literally `"0.0.0-mock"`. Nothing here decodes real pixels or real audio
samples -- there is no real footage with RGB/string-mask/audio ground
truth in this repository yet (same situation Prompt B/C's mocks were
written to cover before MediaPipe/PyTorch adapters existed).
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from yoyovision_ml.adapters_registry import (
    register_audio_analyzer,
    register_rgb_encoder,
    register_string_segmenter,
)
from yoyovision_ml.domain import FeatureFrame, FeatureSet, Track
from yoyovision_ml.interfaces import FrameRef
from yoyovision_ml.multimodal.features import (
    AUDIO_FEATURE_NAMES,
    FEATURE_AUDIO_BEAT_PHASE,
    FEATURE_AUDIO_ONSET_STRENGTH,
    FEATURE_AUDIO_RMS_ENERGY,
    FEATURE_AUDIO_TEMPO_BPM,
    FEATURE_RGB_BRIGHTNESS_MEAN,
    FEATURE_RGB_EMBED_0,
    FEATURE_RGB_EMBED_1,
    FEATURE_RGB_EMBED_2,
    FEATURE_RGB_EMBED_3,
    FEATURE_RGB_SCENE_MOTION_SCORE,
    FEATURE_STRING_SEG_ANGLE_DEG,
    FEATURE_STRING_SEG_CONFIDENCE,
    FEATURE_STRING_SEG_SLACK_ESTIMATE,
    FEATURE_STRING_SEG_VISIBLE_RATIO,
    RGB_FEATURE_NAMES,
    STRING_SEGMENTATION_FEATURE_NAMES,
)

#: Fixed sample rate a mock audio analyzer invents for a clip, independent
#: of the actual video fps -- mirrors `adapters_mock._MOCK_FPS_DEFAULT`'s
#: "invent a plausible timeline" approach for a modality this repository
#: has no real decoder for yet.
_MOCK_AUDIO_SAMPLE_FPS = 10.0


def _stable_seed(*parts: str) -> int:
    """Same deterministic 64-bit seed helper as the package-root
    `adapters_mock._stable_seed` -- duplicated rather than imported so this
    module has no import-time dependency on the perception-mock module."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int(struct.unpack(">Q", digest[:8])[0])


def _deterministic_unit_floats(seed: int, count: int) -> list[float]:
    """Same tiny xorshift PRNG as `adapters_mock._deterministic_unit_floats`."""
    values: list[float] = []
    state = seed or 1
    for _ in range(count):
        state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 7
        state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
        values.append((state % 10_000) / 10_000.0)
    return values


def _bbox_center(track: Track) -> tuple[float, float]:
    return (track.bbox.x + track.bbox.width / 2.0, track.bbox.y + track.bbox.height / 2.0)


@register_rgb_encoder("mock")
class MockRgbEncoder:
    """Deterministic mock RGB/appearance encoder. NOT a trained model.

    Ignores `FrameRef.array` entirely (real pixel content) -- seeds purely
    from `frame_ms`, same "same input -> same output" determinism as every
    other mock adapter in this repository.
    """

    model_name = "mock-rgb-encoder"
    model_version = "0.0.0-mock"

    def encode(self, frame_batch: list[FrameRef]) -> FeatureSet:
        frames: list[FeatureFrame] = []
        for frame in frame_batch:
            seed = _stable_seed("rgb", str(frame.frame_ms))
            jitter = _deterministic_unit_floats(seed, 6)
            values = {
                FEATURE_RGB_EMBED_0: 2.0 * jitter[0] - 1.0,
                FEATURE_RGB_EMBED_1: 2.0 * jitter[1] - 1.0,
                FEATURE_RGB_EMBED_2: 2.0 * jitter[2] - 1.0,
                FEATURE_RGB_EMBED_3: 2.0 * jitter[3] - 1.0,
                FEATURE_RGB_SCENE_MOTION_SCORE: jitter[4],
                FEATURE_RGB_BRIGHTNESS_MEAN: jitter[5],
            }
            frames.append(FeatureFrame(frame_ms=frame.frame_ms, values=values))
        fps = 0.0
        if len(frame_batch) >= 2:
            span_ms = frame_batch[-1].frame_ms - frame_batch[0].frame_ms
            fps = (len(frame_batch) - 1) * 1000.0 / span_ms if span_ms > 0 else 0.0
        return FeatureSet(frames=tuple(frames), feature_names=RGB_FEATURE_NAMES, fps=fps)


@register_string_segmenter("mock")
class MockStringSegmenter:
    """Deterministic mock pixel-based string segmenter. NOT a trained model.

    Ignores `FrameRef.array`; seeds from `frame_ms` plus the matching
    track's bbox center so output is still *somewhat* track-informed (a
    segmenter with zero relationship to where the yo-yo actually is would
    be a strange mock to fuse into the pipeline), while remaining a
    fully-synthetic heuristic, not a real segmentation model.
    """

    model_name = "mock-string-segmenter"
    model_version = "0.0.0-mock"

    def segment(self, frame_batch: list[FrameRef], yoyo_track: list[Track]) -> FeatureSet:
        track_by_ms = {track.frame_ms: track for track in yoyo_track}
        frames: list[FeatureFrame] = []
        for frame in frame_batch:
            track = track_by_ms.get(frame.frame_ms)
            if track is None:
                continue
            center = _bbox_center(track)
            seed = _stable_seed("string-seg", str(frame.frame_ms), f"{center[0]:.4f}")
            jitter = _deterministic_unit_floats(seed, 4)
            values = {
                FEATURE_STRING_SEG_VISIBLE_RATIO: jitter[0],
                FEATURE_STRING_SEG_ANGLE_DEG: jitter[1] * 360.0,
                FEATURE_STRING_SEG_SLACK_ESTIMATE: jitter[2],
                FEATURE_STRING_SEG_CONFIDENCE: 0.4 + jitter[3] * 0.5,
            }
            frames.append(FeatureFrame(frame_ms=frame.frame_ms, values=values))
        fps = 0.0
        if len(frames) >= 2:
            span_ms = frames[-1].frame_ms - frames[0].frame_ms
            fps = (len(frames) - 1) * 1000.0 / span_ms if span_ms > 0 else 0.0
        return FeatureSet(
            frames=tuple(frames), feature_names=STRING_SEGMENTATION_FEATURE_NAMES, fps=fps
        )


@register_audio_analyzer("mock")
class MockAudioAnalyzer:
    """Deterministic mock audio analyzer. NOT a trained model, and does not
    decode any real audio track -- there is no audio-bearing sample footage
    in this repository yet (Prompt A's sample dataset ships placeholder
    video bytes, see `ml/scripts/generate_sample_dataset.py`). Invents a
    plausible fixed-rate timeline across `duration_ms`, seeded from
    `video_path` so results stay reproducible per clip.
    """

    model_name = "mock-audio-analyzer"
    model_version = "0.0.0-mock"

    def analyze(self, video_path: Path, duration_ms: int) -> FeatureSet:
        if duration_ms <= 0:
            return FeatureSet(frames=(), feature_names=AUDIO_FEATURE_NAMES, fps=0.0)

        fps = _MOCK_AUDIO_SAMPLE_FPS
        n_frames = max(1, int(duration_ms / 1000 * fps))
        frames: list[FeatureFrame] = []
        for frame_idx in range(n_frames):
            frame_ms = int(frame_idx / fps * 1000)
            seed = _stable_seed("audio", str(video_path), str(frame_ms))
            jitter = _deterministic_unit_floats(seed, 4)
            values = {
                FEATURE_AUDIO_ONSET_STRENGTH: jitter[0],
                FEATURE_AUDIO_TEMPO_BPM: 60.0 + jitter[1] * 120.0,
                FEATURE_AUDIO_BEAT_PHASE: jitter[2],
                FEATURE_AUDIO_RMS_ENERGY: jitter[3],
            }
            frames.append(FeatureFrame(frame_ms=frame_ms, values=values))
        return FeatureSet(frames=tuple(frames), feature_names=AUDIO_FEATURE_NAMES, fps=fps)
