"""String/slack geometry analysis.

This is real, deterministic geometric computation (not a mock stub): given a
tracked yo-yo position and hand landmarks (which may themselves currently
come from mock upstream adapters), it computes a normalized hand-to-yoyo
distance signal per frame -- a reasonable proxy feature for slack/string
behaviour until a dedicated string-tracking model is available.
"""

from __future__ import annotations

import math

from yoyovision_ml.domain import FeatureFrame, FeatureSet, HandSequence, Track

FEATURE_HAND_YOYO_DISTANCE = "hand_yoyo_distance"
FEATURE_HAND_YOYO_DISTANCE_DELTA = "hand_yoyo_distance_delta"


def _bbox_center(track: Track) -> tuple[float, float]:
    return (track.bbox.x + track.bbox.width / 2.0, track.bbox.y + track.bbox.height / 2.0)


def _nearest_hand_center(hand_sequence: HandSequence, frame_ms: int) -> tuple[float, float] | None:
    candidates = [f for f in hand_sequence.frames if f.frame_ms == frame_ms]
    if not candidates:
        return None
    xs = [kp.x for frame in candidates for kp in frame.keypoints]
    ys = [kp.y for frame in candidates for kp in frame.keypoints]
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


class DeterministicStringAnalyzer:
    """Default implementation of the `StringAnalyzer` protocol."""

    model_name = "deterministic-string-analyzer"
    model_version = "0.1.0"

    def analyze(self, yoyo_track: list[Track], hand_sequence: HandSequence) -> FeatureSet:
        frames: list[FeatureFrame] = []
        previous_distance: float | None = None

        for track in sorted(yoyo_track, key=lambda t: t.frame_ms):
            yoyo_center = _bbox_center(track)
            hand_center = _nearest_hand_center(hand_sequence, track.frame_ms)
            if hand_center is None:
                continue
            distance = math.hypot(yoyo_center[0] - hand_center[0], yoyo_center[1] - hand_center[1])
            delta = 0.0 if previous_distance is None else distance - previous_distance
            previous_distance = distance
            frames.append(
                FeatureFrame(
                    frame_ms=track.frame_ms,
                    values={
                        FEATURE_HAND_YOYO_DISTANCE: distance,
                        FEATURE_HAND_YOYO_DISTANCE_DELTA: delta,
                    },
                )
            )

        fps = hand_sequence.fps if hand_sequence.frames else 0.0
        return FeatureSet(
            frames=tuple(frames),
            feature_names=(FEATURE_HAND_YOYO_DISTANCE, FEATURE_HAND_YOYO_DISTANCE_DELTA),
            fps=fps,
        )
