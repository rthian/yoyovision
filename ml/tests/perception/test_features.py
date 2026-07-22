from __future__ import annotations

import math

from yoyovision_ml.domain import (
    BoundingBox,
    HandFrame,
    HandSequence,
    Keypoint,
    PoseFrame,
    PoseSequence,
    Track,
    VisibilityState,
)
from yoyovision_ml.perception.features import (
    ALL_FEATURE_NAMES,
    FEATURE_HAND_DISTANCE,
    FEATURE_LEFT_ELBOW_ANGLE_DEG,
    FEATURE_SHOULDER_WIDTH,
    FEATURE_YOYO_ACCELERATION,
    FEATURE_YOYO_CONFIDENCE,
    FEATURE_YOYO_INTERPOLATED,
    FEATURE_YOYO_REL_LEFT_WRIST_X,
    FEATURE_YOYO_VELOCITY,
    FEATURE_YOYO_VISIBILITY_CODE,
    FEATURE_YOYO_X,
    FEATURE_YOYO_Y,
    VISIBILITY_CODE,
    compute_kinematic_features,
)


def _track(frame_ms: int, cx: float, cy: float, confidence: float = 0.9) -> Track:
    width = height = 0.05
    return Track(
        track_id="track-0",
        frame_ms=frame_ms,
        bbox=BoundingBox(x=cx - width / 2.0, y=cy - height / 2.0, width=width, height=height),
        confidence=confidence,
        class_label="yoyo",
    )


def _pose_frame(frame_ms: int) -> PoseFrame:
    keypoints = (
        Keypoint(name="left_shoulder", x=0.4, y=0.3, z=0.0, visibility=1.0),
        Keypoint(name="right_shoulder", x=0.6, y=0.3, z=0.0, visibility=1.0),
        Keypoint(name="left_elbow", x=0.35, y=0.5, z=0.0, visibility=1.0),
        Keypoint(name="right_elbow", x=0.65, y=0.5, z=0.0, visibility=1.0),
        Keypoint(name="left_hip", x=0.45, y=0.7, z=0.0, visibility=1.0),
        Keypoint(name="right_hip", x=0.55, y=0.7, z=0.0, visibility=1.0),
    )
    return PoseFrame(frame_ms=frame_ms, keypoints=keypoints, confidence=0.9)


def _hand_frame(frame_ms: int, handedness: str, wrist_x: float, wrist_y: float) -> HandFrame:
    keypoints = (Keypoint(name="wrist", x=wrist_x, y=wrist_y, z=0.0, visibility=1.0),)
    return HandFrame(frame_ms=frame_ms, handedness=handedness, keypoints=keypoints, confidence=0.9)


def test_empty_track_list_produces_empty_feature_set() -> None:
    pose_sequence = PoseSequence(frames=(), model_name="m", model_version="0", fps=30.0)
    hand_sequence = HandSequence(frames=(), model_name="m", model_version="0", fps=30.0)
    feature_set = compute_kinematic_features(pose_sequence, hand_sequence, [], fps=15.0)

    assert feature_set.frames == ()
    assert feature_set.feature_names == ALL_FEATURE_NAMES
    assert feature_set.fps == 15.0


def test_single_frame_has_zero_velocity_and_acceleration() -> None:
    pose_sequence = PoseSequence(
        frames=(_pose_frame(0),), model_name="m", model_version="0", fps=30.0
    )
    hand_sequence = HandSequence(frames=(), model_name="m", model_version="0", fps=30.0)
    tracks = [_track(0, 0.5, 0.5)]

    feature_set = compute_kinematic_features(pose_sequence, hand_sequence, tracks, fps=15.0)

    assert len(feature_set.frames) == 1
    values = feature_set.frames[0].values
    assert values[FEATURE_YOYO_X] == 0.5
    assert values[FEATURE_YOYO_Y] == 0.5
    assert values[FEATURE_YOYO_VELOCITY] == 0.0
    assert values[FEATURE_YOYO_ACCELERATION] == 0.0


def test_velocity_matches_finite_difference_over_real_timestamps() -> None:
    pose_sequence = PoseSequence(frames=(), model_name="m", model_version="0", fps=30.0)
    hand_sequence = HandSequence(frames=(), model_name="m", model_version="0", fps=30.0)
    # Moves from x=0.1 to x=0.3 over 200ms -> velocity magnitude = 0.2 / 0.2s = 1.0.
    tracks = [_track(0, 0.1, 0.5), _track(200, 0.3, 0.5)]

    feature_set = compute_kinematic_features(pose_sequence, hand_sequence, tracks, fps=15.0)

    second = feature_set.frames[1].values
    assert math.isclose(second[FEATURE_YOYO_VELOCITY], 1.0, rel_tol=1e-6)


def test_shoulder_width_and_elbow_angle_computed_from_pose() -> None:
    pose_sequence = PoseSequence(
        frames=(_pose_frame(0),), model_name="m", model_version="0", fps=30.0
    )
    hand_sequence = HandSequence(frames=(), model_name="m", model_version="0", fps=30.0)
    tracks = [_track(0, 0.5, 0.5)]

    feature_set = compute_kinematic_features(pose_sequence, hand_sequence, tracks, fps=15.0)
    values = feature_set.frames[0].values

    expected_shoulder_width = math.hypot(0.4 - 0.6, 0.3 - 0.3)
    assert math.isclose(values[FEATURE_SHOULDER_WIDTH], expected_shoulder_width)
    # Left wrist is missing from the pose fixture, so elbow angle defaults to 0.0.
    assert values[FEATURE_LEFT_ELBOW_ANGLE_DEG] == 0.0


def test_hand_distance_and_relative_wrist_offset_from_hand_sequence() -> None:
    pose_sequence = PoseSequence(frames=(), model_name="m", model_version="0", fps=30.0)
    hand_sequence = HandSequence(
        frames=(_hand_frame(0, "left", 0.3, 0.5), _hand_frame(0, "right", 0.7, 0.5)),
        model_name="m",
        model_version="0",
        fps=30.0,
    )
    tracks = [_track(0, 0.5, 0.5)]

    feature_set = compute_kinematic_features(pose_sequence, hand_sequence, tracks, fps=15.0)
    values = feature_set.frames[0].values

    assert math.isclose(values[FEATURE_HAND_DISTANCE], 0.4)
    assert math.isclose(values[FEATURE_YOYO_REL_LEFT_WRIST_X], 0.5 - 0.3)


def test_visibility_and_interpolated_flags_propagate_into_feature_values() -> None:
    pose_sequence = PoseSequence(frames=(), model_name="m", model_version="0", fps=30.0)
    hand_sequence = HandSequence(frames=(), model_name="m", model_version="0", fps=30.0)
    track = Track(
        track_id="track-0",
        frame_ms=0,
        bbox=BoundingBox(x=0.5, y=0.5, width=0.05, height=0.05),
        confidence=0.42,
        class_label="yoyo",
        visibility=VisibilityState.FULLY_OCCLUDED,
        interpolated=True,
    )

    feature_set = compute_kinematic_features(pose_sequence, hand_sequence, [track], fps=15.0)
    values = feature_set.frames[0].values

    assert values[FEATURE_YOYO_CONFIDENCE] == 0.42
    assert values[FEATURE_YOYO_VISIBILITY_CODE] == VISIBILITY_CODE[VisibilityState.FULLY_OCCLUDED]
    assert values[FEATURE_YOYO_INTERPOLATED] == 1.0


def test_tracks_are_processed_in_frame_ms_order_regardless_of_input_order() -> None:
    pose_sequence = PoseSequence(frames=(), model_name="m", model_version="0", fps=30.0)
    hand_sequence = HandSequence(frames=(), model_name="m", model_version="0", fps=30.0)
    # Deliberately out of order.
    tracks = [_track(200, 0.5, 0.5), _track(0, 0.1, 0.5), _track(100, 0.3, 0.5)]

    feature_set = compute_kinematic_features(pose_sequence, hand_sequence, tracks, fps=15.0)

    assert [f.frame_ms for f in feature_set.frames] == [0, 100, 200]
