"""Combines pose/hand/yo-yo-track/string signals into one per-frame feature timeline.

This is real, deterministic aggregation logic (not a mock): it merges
whatever upstream sequences it is given (currently produced by mock
adapters, later real ones) into a single indexed `FeatureSet` keyed by
frame_ms, which is what `TemporalEventDetector` implementations consume.
"""

from __future__ import annotations

import math

from yoyovision_ml.domain import FeatureFrame, FeatureSet, HandSequence, PoseSequence, Track

FEATURE_POSE_MOTION = "pose_motion_magnitude"
FEATURE_HAND_SPAN = "hand_span"


def _pose_motion_magnitude(
    previous: PoseSequence | None, current_idx: int, pose_sequence: PoseSequence
) -> float:
    if current_idx == 0:
        return 0.0
    current = pose_sequence.frames[current_idx]
    previous_frame = pose_sequence.frames[current_idx - 1]
    total = 0.0
    for kp_now, kp_prev in zip(current.keypoints, previous_frame.keypoints, strict=False):
        total += math.hypot(kp_now.x - kp_prev.x, kp_now.y - kp_prev.y)
    return total / max(1, len(current.keypoints))


def _hand_span(hand_sequence: HandSequence, frame_ms: int) -> float:
    frames = [f for f in hand_sequence.frames if f.frame_ms == frame_ms]
    if len(frames) < 2:
        return 0.0
    left = next((f for f in frames if f.handedness == "left"), None)
    right = next((f for f in frames if f.handedness == "right"), None)
    if left is None or right is None or not left.keypoints or not right.keypoints:
        return 0.0
    lx = sum(kp.x for kp in left.keypoints) / len(left.keypoints)
    ly = sum(kp.y for kp in left.keypoints) / len(left.keypoints)
    rx = sum(kp.x for kp in right.keypoints) / len(right.keypoints)
    ry = sum(kp.y for kp in right.keypoints) / len(right.keypoints)
    return math.hypot(lx - rx, ly - ry)


class DeterministicFeatureExtractor:
    """Default implementation of the `FeatureExtractor` protocol."""

    def extract(
        self,
        pose_sequence: PoseSequence,
        hand_sequence: HandSequence,
        yoyo_tracks: list[Track],
        string_features: FeatureSet,
    ) -> FeatureSet:
        string_by_frame = {f.frame_ms: f.values for f in string_features.frames}
        merged: list[FeatureFrame] = []

        for idx, pose_frame in enumerate(pose_sequence.frames):
            values: dict[str, float] = {
                FEATURE_POSE_MOTION: _pose_motion_magnitude(None, idx, pose_sequence),
                FEATURE_HAND_SPAN: _hand_span(hand_sequence, pose_frame.frame_ms),
            }
            values.update(string_by_frame.get(pose_frame.frame_ms, {}))
            merged.append(FeatureFrame(frame_ms=pose_frame.frame_ms, values=values))

        observed_names = {name for frame in merged for name in frame.values}
        feature_names = tuple(sorted(observed_names | {FEATURE_POSE_MOTION, FEATURE_HAND_SPAN}))
        return FeatureSet(frames=tuple(merged), feature_names=feature_names, fps=pose_sequence.fps)
