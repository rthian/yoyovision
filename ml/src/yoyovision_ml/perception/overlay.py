"""Debug overlay-video generation (Prompt B): renders the yo-yo track (color-
coded by confidence, visually distinct when interpolated), hand keypoints,
and shoulder/elbow/wrist pose keypoints on top of the source video.

Requires OpenCV (the `mediapipe` extra brings `opencv-python-headless`, or
it can be installed standalone) -- raises `MissingOptionalDependencyError`
immediately and clearly if unavailable, exactly like the other optional
perception adapters.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any

from yoyovision_ml.domain import HandSequence, PoseSequence, Track
from yoyovision_ml.perception.errors import MissingOptionalDependencyError

#: BGR (OpenCV convention).
_COLOR_HIGH_CONFIDENCE = (0, 200, 0)
_COLOR_LOW_CONFIDENCE = (0, 128, 255)
_COLOR_INTERPOLATED = (0, 0, 220)
_COLOR_POSE = (255, 255, 0)
_COLOR_HAND_LEFT = (255, 0, 255)
_COLOR_HAND_RIGHT = (0, 255, 255)
_CONFIDENCE_COLOR_THRESHOLD = 0.5
_POSE_JOINTS_DRAWN = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise MissingOptionalDependencyError("cv2", "mediapipe") from exc
    return cv2


def _nearest_ms(sorted_ms: list[int], target_ms: int, tolerance_ms: int = 200) -> int | None:
    if not sorted_ms:
        return None
    idx = bisect.bisect_left(sorted_ms, target_ms)
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(sorted_ms)]
    if not candidates:
        return None
    nearest_idx = min(candidates, key=lambda i: abs(sorted_ms[i] - target_ms))
    nearest = sorted_ms[nearest_idx]
    return nearest if abs(nearest - target_ms) <= tolerance_ms else None


def render_overlay_video(
    video_path: Path,
    output_path: Path,
    tracks: list[Track],
    pose_sequence: PoseSequence | None = None,
    hand_sequence: HandSequence | None = None,
) -> Path:
    """Writes an annotated copy of `video_path` to `output_path` (same
    container/fps as the source) for visual debugging of detection
    confidence, interpolated frames, and pose/hand landmarks.
    """
    cv2 = _import_cv2()

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    tracks_by_ms = {t.frame_ms: t for t in tracks}
    track_ms_sorted = sorted(tracks_by_ms)
    pose_by_ms = {f.frame_ms: f for f in (pose_sequence.frames if pose_sequence else ())}
    pose_ms_sorted = sorted(pose_by_ms)
    hand_frames = hand_sequence.frames if hand_sequence else ()

    try:
        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_ms = int(round(frame_idx / fps * 1000))

            nearest_track_ms = _nearest_ms(track_ms_sorted, frame_ms)
            if nearest_track_ms is not None:
                _draw_track(cv2, frame, tracks_by_ms[nearest_track_ms], width, height)

            nearest_pose_ms = _nearest_ms(pose_ms_sorted, frame_ms)
            if nearest_pose_ms is not None:
                _draw_pose(cv2, frame, pose_by_ms[nearest_pose_ms], width, height)

            for hand_frame in hand_frames:
                if abs(hand_frame.frame_ms - frame_ms) <= 100:
                    _draw_hand(cv2, frame, hand_frame, width, height)

            _draw_frame_label(cv2, frame, frame_ms)
            writer.write(frame)
            frame_idx += 1
    finally:
        capture.release()
        writer.release()

    return output_path


def _denorm(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return int(round(x * width)), int(round(y * height))


def _draw_track(cv2_module: Any, frame: Any, track: Track, width: int, height: int) -> None:
    x1, y1 = _denorm(track.bbox.x, track.bbox.y, width, height)
    x2, y2 = _denorm(
        track.bbox.x + track.bbox.width, track.bbox.y + track.bbox.height, width, height
    )
    if track.interpolated:
        color = _COLOR_INTERPOLATED
    elif track.confidence >= _CONFIDENCE_COLOR_THRESHOLD:
        color = _COLOR_HIGH_CONFIDENCE
    else:
        color = _COLOR_LOW_CONFIDENCE
    line_type = cv2_module.LINE_4 if track.interpolated else cv2_module.LINE_AA
    cv2_module.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=line_type)
    label = f"{track.confidence:.2f}{' (interp)' if track.interpolated else ''}"
    cv2_module.putText(
        frame, label, (x1, max(0, y1 - 6)), cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
    )


def _draw_pose(cv2_module: Any, frame: Any, pose_frame: Any, width: int, height: int) -> None:
    for kp in pose_frame.keypoints:
        if kp.name not in _POSE_JOINTS_DRAWN:
            continue
        point = _denorm(kp.x, kp.y, width, height)
        cv2_module.circle(frame, point, 4, _COLOR_POSE, -1)


def _draw_hand(cv2_module: Any, frame: Any, hand_frame: Any, width: int, height: int) -> None:
    color = _COLOR_HAND_LEFT if hand_frame.handedness == "left" else _COLOR_HAND_RIGHT
    for kp in hand_frame.keypoints:
        point = _denorm(kp.x, kp.y, width, height)
        cv2_module.circle(frame, point, 2, color, -1)


def _draw_frame_label(cv2_module: Any, frame: Any, frame_ms: int) -> None:
    text = f"t={frame_ms}ms"
    cv2_module.putText(
        frame, text, (10, 20), cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
