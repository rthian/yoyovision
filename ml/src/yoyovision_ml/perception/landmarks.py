"""Canonical pose/hand landmark topology, shared by every pose/hand backend.

Both the deterministic mock adapters (`adapters_mock.py`) and the real
MediaPipe adapters (`perception/detector_mediapipe.py`) must agree on landmark
*names* and their *order*, because `perception/features.py` looks up specific
joints (wrists, elbows, shoulders) by name to compute kinematic features
regardless of which backend produced the `PoseSequence`/`HandSequence`.

This module has zero dependencies beyond the standard library so it can be
imported from `adapters_mock.py` without pulling in anything from the
`perception` package's heavier (optional-dependency) submodules.

The 33-point body topology and 21-point hand topology below match MediaPipe's
BlazePose/Hand landmark ordering (https://developers.google.com/mediapipe),
which is also what a real MediaPipe adapter emits -- but the names themselves
are the contract callers rely on, not the specific upstream model.
"""

from __future__ import annotations

#: MediaPipe BlazePose full-body topology, index-ordered (33 points).
POSE_LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

#: MediaPipe Hand topology, index-ordered (21 points), same names used for
#: both the "left" and "right" `HandFrame.handedness` sequences.
HAND_LANDMARK_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

if len(POSE_LANDMARK_NAMES) != 33:  # pragma: no cover - topology invariant
    raise AssertionError("POSE_LANDMARK_NAMES must have exactly 33 entries")
if len(HAND_LANDMARK_NAMES) != 21:  # pragma: no cover - topology invariant
    raise AssertionError("HAND_LANDMARK_NAMES must have exactly 21 entries")
