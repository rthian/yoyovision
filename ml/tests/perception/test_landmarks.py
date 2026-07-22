from __future__ import annotations

from yoyovision_ml.perception.landmarks import HAND_LANDMARK_NAMES, POSE_LANDMARK_NAMES


def test_pose_landmark_names_has_33_unique_entries() -> None:
    assert len(POSE_LANDMARK_NAMES) == 33
    assert len(set(POSE_LANDMARK_NAMES)) == 33


def test_hand_landmark_names_has_21_unique_entries() -> None:
    assert len(HAND_LANDMARK_NAMES) == 21
    assert len(set(HAND_LANDMARK_NAMES)) == 21


def test_pose_landmark_names_include_arm_joints_used_by_features() -> None:
    required = {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
    }
    assert required.issubset(set(POSE_LANDMARK_NAMES))


def test_hand_landmark_names_include_wrist() -> None:
    assert "wrist" in HAND_LANDMARK_NAMES
