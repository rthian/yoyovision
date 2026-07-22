"""Real MediaPipe pose/hand adapters (`PoseEstimator`/`HandEstimator` protocols).

Unlike the yo-yo detector, MediaPipe ships ready-to-use pretrained pose/hand
landmark models inside the `mediapipe` pip package itself -- no separate
checkpoint needs to be configured, so this is genuine, working inference the
moment `pip install 'yoyovision-ml[mediapipe]'` has been run, per Prompt B's
"Use MediaPipe through adapters for the initial pose and hand implementation."

Uses the classic `mediapipe.solutions.pose`/`mediapipe.solutions.hands` API
(bundled model assets, no `.task` file download required) rather than the
newer Tasks API, specifically to avoid needing any additional configured
model file for what MediaPipe itself treats as an off-the-shelf model.

Landmark names/order are normalized to `perception.landmarks` so
`perception.features` can treat mock and MediaPipe pose/hand sequences
identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yoyovision_ml.adapters_registry import register_hand_estimator, register_pose_estimator
from yoyovision_ml.domain import HandFrame, HandSequence, Keypoint, PoseFrame, PoseSequence
from yoyovision_ml.perception.errors import MissingOptionalDependencyError
from yoyovision_ml.perception.landmarks import HAND_LANDMARK_NAMES, POSE_LANDMARK_NAMES


def _import_mediapipe_and_cv2() -> tuple[Any, Any]:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise MissingOptionalDependencyError("mediapipe", "mediapipe") from exc
    try:
        import cv2
    except ImportError as exc:
        raise MissingOptionalDependencyError("cv2", "mediapipe") from exc
    return mp, cv2


def _iter_video_frames(cv2_module: Any, video_path: Path) -> Any:
    """Yields `(frame_ms, bgr_frame, fps)` using the video's own reported
    fps, consistent with `preprocessing.py`'s "never re-base the timeline" rule.
    """
    capture = cv2_module.VideoCapture(str(video_path))
    fps = capture.get(cv2_module.CAP_PROP_FPS) or 30.0
    try:
        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield int(round(frame_idx / fps * 1000)), frame, fps
            frame_idx += 1
    finally:
        capture.release()


@register_pose_estimator("mediapipe")
class MediaPipePoseEstimator:
    """Real body-pose estimator backed by MediaPipe's bundled BlazePose model."""

    model_name = "mediapipe-pose-estimator"

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5) -> None:
        self._model_complexity = model_complexity
        self._min_detection_confidence = min_detection_confidence
        mp, _ = _import_mediapipe_and_cv2()
        self.model_version = (
            f"solutions-pose@complexity{model_complexity}+mediapipe{mp.__version__}"
        )

    def predict(self, video_path: Path) -> PoseSequence:
        mp, cv2 = _import_mediapipe_and_cv2()
        frames: list[PoseFrame] = []
        fps_seen = 30.0

        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self._model_complexity,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_detection_confidence,
        ) as pose:
            for frame_ms, bgr_frame, fps in _iter_video_frames(cv2, video_path):
                fps_seen = fps
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb_frame)
                if result.pose_landmarks is None:
                    continue
                keypoints = tuple(
                    Keypoint(name=name, x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
                    for name, lm in zip(
                        POSE_LANDMARK_NAMES, result.pose_landmarks.landmark, strict=True
                    )
                )
                confidence = sum(kp.visibility for kp in keypoints) / len(keypoints)
                frames.append(
                    PoseFrame(frame_ms=frame_ms, keypoints=keypoints, confidence=confidence)
                )

        return PoseSequence(
            frames=tuple(frames),
            model_name=self.model_name,
            model_version=self.model_version,
            fps=fps_seen,
        )


@register_hand_estimator("mediapipe")
class MediaPipeHandEstimator:
    """Real hand-landmark estimator backed by MediaPipe's bundled hand model."""

    model_name = "mediapipe-hand-estimator"

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self._min_detection_confidence = min_detection_confidence
        mp, _ = _import_mediapipe_and_cv2()
        self.model_version = f"solutions-hands+mediapipe{mp.__version__}"

    def predict(self, video_path: Path) -> HandSequence:
        mp, cv2 = _import_mediapipe_and_cv2()
        frames: list[HandFrame] = []
        fps_seen = 30.0

        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_detection_confidence,
        ) as hands:
            for frame_ms, bgr_frame, fps in _iter_video_frames(cv2, video_path):
                fps_seen = fps
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)
                if not result.multi_hand_landmarks:
                    continue
                handedness_list = result.multi_handedness or []
                for hand_landmarks, handedness in zip(
                    result.multi_hand_landmarks, handedness_list, strict=False
                ):
                    classification = handedness.classification[0]
                    # MediaPipe reports the hand as seen by the camera (mirrored);
                    # `.lower()` matches this project's "left"/"right" convention.
                    label = classification.label.lower()
                    keypoints = tuple(
                        Keypoint(name=name, x=lm.x, y=lm.y, z=lm.z, visibility=classification.score)
                        for name, lm in zip(
                            HAND_LANDMARK_NAMES, hand_landmarks.landmark, strict=True
                        )
                    )
                    frames.append(
                        HandFrame(
                            frame_ms=frame_ms,
                            handedness=label,
                            keypoints=keypoints,
                            confidence=classification.score,
                        )
                    )

        return HandSequence(
            frames=tuple(frames),
            model_name=self.model_name,
            model_version=self.model_version,
            fps=fps_seen,
        )
