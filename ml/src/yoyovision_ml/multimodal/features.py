"""Feature-column names for Prompt E's RGB/string-segmentation/audio
adapters, plus `fuse_feature_sets` -- the real, deterministic (not mock)
merge step that combines those modality `FeatureSet`s with the existing
kinematic `FeatureSet` (Prompt B/D's pose/hand/track/string-geometry
output) into one enriched per-frame timeline.

Column naming mirrors `perception/features.py`'s convention: one module
constant per column plus an `ALL_*_FEATURE_NAMES` tuple recording the fixed
order. `MULTIMODAL_FEATURE_NAMES` is the concatenation of all three
modalities' columns -- this is what `events/labels.py`'s
`"multimodal_fused"` feature subset selects on top of the existing
kinematic columns.
"""

from __future__ import annotations

import bisect

from yoyovision_ml.domain import FeatureFrame, FeatureSet

# --------------------------------------------------------------------------- #
# RGB (appearance) columns
# --------------------------------------------------------------------------- #
FEATURE_RGB_EMBED_0 = "rgb_embed_0"
FEATURE_RGB_EMBED_1 = "rgb_embed_1"
FEATURE_RGB_EMBED_2 = "rgb_embed_2"
FEATURE_RGB_EMBED_3 = "rgb_embed_3"
FEATURE_RGB_SCENE_MOTION_SCORE = "rgb_scene_motion_score"
FEATURE_RGB_BRIGHTNESS_MEAN = "rgb_brightness_mean"

RGB_FEATURE_NAMES: tuple[str, ...] = (
    FEATURE_RGB_EMBED_0,
    FEATURE_RGB_EMBED_1,
    FEATURE_RGB_EMBED_2,
    FEATURE_RGB_EMBED_3,
    FEATURE_RGB_SCENE_MOTION_SCORE,
    FEATURE_RGB_BRIGHTNESS_MEAN,
)

# --------------------------------------------------------------------------- #
# String segmentation columns
# --------------------------------------------------------------------------- #
FEATURE_STRING_SEG_VISIBLE_RATIO = "string_seg_visible_ratio"
FEATURE_STRING_SEG_ANGLE_DEG = "string_seg_angle_deg"
FEATURE_STRING_SEG_SLACK_ESTIMATE = "string_seg_slack_estimate"
FEATURE_STRING_SEG_CONFIDENCE = "string_seg_confidence"

STRING_SEGMENTATION_FEATURE_NAMES: tuple[str, ...] = (
    FEATURE_STRING_SEG_VISIBLE_RATIO,
    FEATURE_STRING_SEG_ANGLE_DEG,
    FEATURE_STRING_SEG_SLACK_ESTIMATE,
    FEATURE_STRING_SEG_CONFIDENCE,
)

# --------------------------------------------------------------------------- #
# Audio columns
# --------------------------------------------------------------------------- #
FEATURE_AUDIO_ONSET_STRENGTH = "audio_onset_strength"
FEATURE_AUDIO_TEMPO_BPM = "audio_tempo_bpm"
FEATURE_AUDIO_BEAT_PHASE = "audio_beat_phase"
FEATURE_AUDIO_RMS_ENERGY = "audio_rms_energy"

AUDIO_FEATURE_NAMES: tuple[str, ...] = (
    FEATURE_AUDIO_ONSET_STRENGTH,
    FEATURE_AUDIO_TEMPO_BPM,
    FEATURE_AUDIO_BEAT_PHASE,
    FEATURE_AUDIO_RMS_ENERGY,
)

#: Every Prompt E column, in a fixed order -- what `events/labels.py`'s
#: `"multimodal_fused"` subset adds on top of the kinematic columns.
MULTIMODAL_FEATURE_NAMES: tuple[str, ...] = (
    RGB_FEATURE_NAMES + STRING_SEGMENTATION_FEATURE_NAMES + AUDIO_FEATURE_NAMES
)


def _nearest_frame_values(
    sorted_frame_ms: list[int], values_by_ms: dict[int, dict[str, float]], target_ms: int
) -> dict[str, float]:
    """Nearest-neighbor lookup, same pattern as
    `perception/features.py::_nearest_pose_frame` -- the RGB/string-seg/audio
    adapters need not sample on exactly the kinematic pipeline's frame grid."""
    if not sorted_frame_ms:
        return {}
    idx = bisect.bisect_left(sorted_frame_ms, target_ms)
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(sorted_frame_ms)]
    if not candidates:
        return {}
    nearest_idx = min(candidates, key=lambda i: abs(sorted_frame_ms[i] - target_ms))
    return values_by_ms[sorted_frame_ms[nearest_idx]]


def fuse_feature_sets(
    kinematic_features: FeatureSet,
    rgb_features: FeatureSet,
    string_seg_features: FeatureSet,
    audio_features: FeatureSet,
) -> FeatureSet:
    """Merges the kinematic `FeatureSet` (from
    `feature_extraction.DeterministicFeatureExtractor`) with the three
    Prompt E modality `FeatureSet`s, keyed to the kinematic timeline's
    frame_ms grid (the pipeline's primary temporal driver, same rationale
    as `perception/features.py::compute_kinematic_features`). Each
    modality's nearest frame (by `frame_ms`) is merged in; a modality with
    no frames at all (e.g. an audio analyzer given a silent/audio-less
    clip) simply contributes no additional columns for that frame rather
    than raising.
    """

    def _index(features: FeatureSet) -> tuple[list[int], dict[int, dict[str, float]]]:
        by_ms = {f.frame_ms: f.values for f in features.frames}
        return sorted(by_ms), by_ms

    rgb_ms, rgb_by_ms = _index(rgb_features)
    string_seg_ms, string_seg_by_ms = _index(string_seg_features)
    audio_ms, audio_by_ms = _index(audio_features)

    merged: list[FeatureFrame] = []
    for frame in kinematic_features.frames:
        values: dict[str, float] = dict(frame.values)
        values.update(_nearest_frame_values(rgb_ms, rgb_by_ms, frame.frame_ms))
        values.update(_nearest_frame_values(string_seg_ms, string_seg_by_ms, frame.frame_ms))
        values.update(_nearest_frame_values(audio_ms, audio_by_ms, frame.frame_ms))
        merged.append(FeatureFrame(frame_ms=frame.frame_ms, values=values))

    observed_names = {name for frame in merged for name in frame.values}
    feature_names = tuple(sorted(observed_names | set(kinematic_features.feature_names)))
    return FeatureSet(
        frames=tuple(merged), feature_names=feature_names, fps=kinematic_features.fps
    )
