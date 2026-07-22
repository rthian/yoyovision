"""Per-timestamp kinematic feature computation for the perception pipeline.

Real, deterministic geometric/kinematic computation (not a mock) over
whatever `PoseSequence`/`HandSequence`/`list[Track]` it is given -- mock or
real backend, it does not care, because both agree on the landmark naming
contract in `perception/landmarks.py`.

All positions are normalized `[0, 1]` image coordinates (origin top-left, x
right, y down) unless otherwise noted -- see `perception/artifact.py`'s
`COORDINATE_CONVENTION` constant, which documents this for the artifact
metadata sidecar. Velocity/acceleration are finite differences over the
*actual* millisecond gap between consecutive yo-yo track frames (not an
assumed constant frame interval), consistent with this project's "preserve
real timestamps, never re-base them" convention (see `preprocessing.py`).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from yoyovision_ml.domain import (
    FeatureFrame,
    FeatureSet,
    HandSequence,
    PoseFrame,
    PoseSequence,
    Track,
    VisibilityState,
)

FEATURE_YOYO_X = "yoyo_x"
FEATURE_YOYO_Y = "yoyo_y"
FEATURE_YOYO_REL_LEFT_WRIST_X = "yoyo_rel_left_wrist_x"
FEATURE_YOYO_REL_LEFT_WRIST_Y = "yoyo_rel_left_wrist_y"
FEATURE_YOYO_REL_RIGHT_WRIST_X = "yoyo_rel_right_wrist_x"
FEATURE_YOYO_REL_RIGHT_WRIST_Y = "yoyo_rel_right_wrist_y"
FEATURE_YOYO_VELOCITY = "yoyo_velocity"
FEATURE_YOYO_ACCELERATION = "yoyo_acceleration"
FEATURE_YOYO_DIRECTION_DEG = "yoyo_direction_deg"
FEATURE_HAND_DISTANCE = "hand_distance"
FEATURE_LEFT_WRIST_VELOCITY = "left_wrist_velocity"
FEATURE_RIGHT_WRIST_VELOCITY = "right_wrist_velocity"
FEATURE_LEFT_ELBOW_ANGLE_DEG = "left_elbow_angle_deg"
FEATURE_RIGHT_ELBOW_ANGLE_DEG = "right_elbow_angle_deg"
FEATURE_SHOULDER_WIDTH = "shoulder_width"
FEATURE_STAGE_X = "stage_x"
FEATURE_STAGE_Y = "stage_y"
FEATURE_YOYO_CONFIDENCE = "yoyo_confidence"
FEATURE_YOYO_VISIBILITY_CODE = "yoyo_visibility_code"
FEATURE_YOYO_INTERPOLATED = "yoyo_interpolated"

#: Ordinal encoding for `FEATURE_YOYO_VISIBILITY_CODE` -- a `FeatureFrame`'s
#: `values` dict is `dict[str, float]`, so the enum must be encoded numerically.
VISIBILITY_CODE: dict[VisibilityState, float] = {
    VisibilityState.VISIBLE: 0.0,
    VisibilityState.PARTIALLY_OCCLUDED: 1.0,
    VisibilityState.FULLY_OCCLUDED: 2.0,
    VisibilityState.OUTSIDE_FRAME: 3.0,
    VisibilityState.UNLABELLED: 4.0,
}

ALL_FEATURE_NAMES: tuple[str, ...] = (
    FEATURE_YOYO_X,
    FEATURE_YOYO_Y,
    FEATURE_YOYO_REL_LEFT_WRIST_X,
    FEATURE_YOYO_REL_LEFT_WRIST_Y,
    FEATURE_YOYO_REL_RIGHT_WRIST_X,
    FEATURE_YOYO_REL_RIGHT_WRIST_Y,
    FEATURE_YOYO_VELOCITY,
    FEATURE_YOYO_ACCELERATION,
    FEATURE_YOYO_DIRECTION_DEG,
    FEATURE_HAND_DISTANCE,
    FEATURE_LEFT_WRIST_VELOCITY,
    FEATURE_RIGHT_WRIST_VELOCITY,
    FEATURE_LEFT_ELBOW_ANGLE_DEG,
    FEATURE_RIGHT_ELBOW_ANGLE_DEG,
    FEATURE_SHOULDER_WIDTH,
    FEATURE_STAGE_X,
    FEATURE_STAGE_Y,
    FEATURE_YOYO_CONFIDENCE,
    FEATURE_YOYO_VISIBILITY_CODE,
    FEATURE_YOYO_INTERPOLATED,
)


def _bbox_center(track: Track) -> tuple[float, float]:
    return (track.bbox.x + track.bbox.width / 2.0, track.bbox.y + track.bbox.height / 2.0)


def _nearest_pose_frame(
    sorted_frame_ms: list[int], frames_by_ms: dict[int, PoseFrame], target_ms: int
) -> PoseFrame | None:
    """Nearest-neighbor lookup: pose/hand sampling need not exactly align
    with the (possibly independently sampled) yo-yo track's frame_ms grid.
    """
    if not sorted_frame_ms:
        return None
    idx = bisect.bisect_left(sorted_frame_ms, target_ms)
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(sorted_frame_ms)]
    if not candidates:
        return None
    nearest_idx = min(candidates, key=lambda i: abs(sorted_frame_ms[i] - target_ms))
    return frames_by_ms[sorted_frame_ms[nearest_idx]]


def _keypoint_xy(frame: PoseFrame | None, name: str) -> tuple[float, float] | None:
    if frame is None:
        return None
    for kp in frame.keypoints:
        if kp.name == name:
            return (kp.x, kp.y)
    return None


def _elbow_angle_deg(
    shoulder: tuple[float, float] | None,
    elbow: tuple[float, float] | None,
    wrist: tuple[float, float] | None,
) -> float | None:
    """Angle at the elbow between the upper arm (shoulder->elbow) and
    forearm (elbow->wrist) vectors, in degrees; 180 deg = fully extended."""
    if shoulder is None or elbow is None or wrist is None:
        return None
    v1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    v2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
    mag1, mag2 = math.hypot(*v1), math.hypot(*v2)
    if mag1 < 1e-9 or mag2 < 1e-9:
        return None
    cos_angle = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


@dataclass
class _HandLandmarks:
    left_wrist: tuple[float, float] | None
    right_wrist: tuple[float, float] | None


def _hand_wrists(hand_sequence: HandSequence, frame_ms: int) -> _HandLandmarks:
    left = right = None
    for frame in hand_sequence.frames:
        if frame.frame_ms != frame_ms or not frame.keypoints:
            continue
        wrist_kp = next((kp for kp in frame.keypoints if kp.name == "wrist"), frame.keypoints[0])
        if frame.handedness == "left":
            left = (wrist_kp.x, wrist_kp.y)
        elif frame.handedness == "right":
            right = (wrist_kp.x, wrist_kp.y)
    return _HandLandmarks(left_wrist=left, right_wrist=right)


def compute_kinematic_features(
    pose_sequence: PoseSequence,
    hand_sequence: HandSequence,
    yoyo_tracks: list[Track],
    fps: float,
) -> FeatureSet:
    """Computes the full Prompt-B kinematic feature list, keyed to each
    yo-yo track frame's timestamp (the primary temporal driver -- almost
    every feature is the yo-yo's position/motion relative to hands/body).
    """
    tracks_sorted = sorted(yoyo_tracks, key=lambda t: t.frame_ms)
    pose_by_ms = {f.frame_ms: f for f in pose_sequence.frames}
    pose_ms_sorted = sorted(pose_by_ms)

    frames: list[FeatureFrame] = []
    prev_center: tuple[float, float] | None = None
    prev_time_s: float | None = None
    prev_velocity: tuple[float, float] | None = None
    prev_left_wrist: tuple[float, float] | None = None
    prev_right_wrist: tuple[float, float] | None = None

    for track in tracks_sorted:
        time_s = track.frame_ms / 1000.0
        center = _bbox_center(track)
        pose_frame = _nearest_pose_frame(pose_ms_sorted, pose_by_ms, track.frame_ms)
        hand_wrists = _hand_wrists(hand_sequence, track.frame_ms)

        left_shoulder = _keypoint_xy(pose_frame, "left_shoulder")
        right_shoulder = _keypoint_xy(pose_frame, "right_shoulder")
        left_elbow = _keypoint_xy(pose_frame, "left_elbow")
        right_elbow = _keypoint_xy(pose_frame, "right_elbow")
        left_hip = _keypoint_xy(pose_frame, "left_hip")
        right_hip = _keypoint_xy(pose_frame, "right_hip")

        dt = None if prev_time_s is None else time_s - prev_time_s
        if dt is not None and dt > 0 and prev_center is not None:
            velocity_vec = ((center[0] - prev_center[0]) / dt, (center[1] - prev_center[1]) / dt)
        else:
            velocity_vec = (0.0, 0.0)
        velocity = math.hypot(*velocity_vec)
        direction_deg = (
            math.degrees(math.atan2(velocity_vec[1], velocity_vec[0])) if velocity > 1e-9 else 0.0
        )

        if dt is not None and dt > 0 and prev_velocity is not None:
            acceleration_vec = (
                (velocity_vec[0] - prev_velocity[0]) / dt,
                (velocity_vec[1] - prev_velocity[1]) / dt,
            )
            acceleration = math.hypot(*acceleration_vec)
        else:
            acceleration = 0.0

        def _wrist_velocity(
            current: tuple[float, float] | None,
            previous: tuple[float, float] | None,
            delta_s: float | None,
        ) -> float:
            if current is None or previous is None or delta_s is None or delta_s <= 0:
                return 0.0
            return math.hypot(
                (current[0] - previous[0]) / delta_s, (current[1] - previous[1]) / delta_s
            )

        shoulder_width = (
            math.hypot(left_shoulder[0] - right_shoulder[0], left_shoulder[1] - right_shoulder[1])
            if left_shoulder is not None and right_shoulder is not None
            else 0.0
        )
        body_center = None
        if left_hip is not None and right_hip is not None:
            body_center = ((left_hip[0] + right_hip[0]) / 2.0, (left_hip[1] + right_hip[1]) / 2.0)
        elif left_shoulder is not None and right_shoulder is not None:
            body_center = (
                (left_shoulder[0] + right_shoulder[0]) / 2.0,
                (left_shoulder[1] + right_shoulder[1]) / 2.0,
            )
        scale = shoulder_width if shoulder_width > 1e-6 else 1.0
        stage_x = (center[0] - body_center[0]) / scale if body_center is not None else 0.0
        stage_y = (center[1] - body_center[1]) / scale if body_center is not None else 0.0

        left_wrist = hand_wrists.left_wrist
        right_wrist = hand_wrists.right_wrist
        hand_distance = (
            math.hypot(left_wrist[0] - right_wrist[0], left_wrist[1] - right_wrist[1])
            if left_wrist is not None and right_wrist is not None
            else 0.0
        )

        left_elbow_angle = _elbow_angle_deg(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = _elbow_angle_deg(right_shoulder, right_elbow, right_wrist)

        values: dict[str, float] = {
            FEATURE_YOYO_X: center[0],
            FEATURE_YOYO_Y: center[1],
            FEATURE_YOYO_REL_LEFT_WRIST_X: (center[0] - left_wrist[0]) if left_wrist else 0.0,
            FEATURE_YOYO_REL_LEFT_WRIST_Y: (center[1] - left_wrist[1]) if left_wrist else 0.0,
            FEATURE_YOYO_REL_RIGHT_WRIST_X: (center[0] - right_wrist[0]) if right_wrist else 0.0,
            FEATURE_YOYO_REL_RIGHT_WRIST_Y: (center[1] - right_wrist[1]) if right_wrist else 0.0,
            FEATURE_YOYO_VELOCITY: velocity,
            FEATURE_YOYO_ACCELERATION: acceleration,
            FEATURE_YOYO_DIRECTION_DEG: direction_deg,
            FEATURE_HAND_DISTANCE: hand_distance,
            FEATURE_LEFT_WRIST_VELOCITY: _wrist_velocity(left_wrist, prev_left_wrist, dt),
            FEATURE_RIGHT_WRIST_VELOCITY: _wrist_velocity(right_wrist, prev_right_wrist, dt),
            FEATURE_LEFT_ELBOW_ANGLE_DEG: left_elbow_angle if left_elbow_angle is not None else 0.0,
            FEATURE_RIGHT_ELBOW_ANGLE_DEG: (
                right_elbow_angle if right_elbow_angle is not None else 0.0
            ),
            FEATURE_SHOULDER_WIDTH: shoulder_width,
            FEATURE_STAGE_X: stage_x,
            FEATURE_STAGE_Y: stage_y,
            FEATURE_YOYO_CONFIDENCE: track.confidence,
            FEATURE_YOYO_VISIBILITY_CODE: VISIBILITY_CODE[track.visibility],
            FEATURE_YOYO_INTERPOLATED: 1.0 if track.interpolated else 0.0,
        }
        frames.append(FeatureFrame(frame_ms=track.frame_ms, values=values))

        prev_center, prev_time_s, prev_velocity = center, time_s, velocity_vec
        prev_left_wrist, prev_right_wrist = left_wrist, right_wrist

    return FeatureSet(frames=tuple(frames), feature_names=ALL_FEATURE_NAMES, fps=fps)
