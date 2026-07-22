"""Frame sampling that preserves original video timestamps (product principle #9).

Every `FrameRef` carries `frame_ms` computed directly from the source video's
own timeline (via ffprobe-derived duration/fps or the video's own frame
index), never a re-numbered/re-based index. This is what lets every
downstream event, evidence reference, and exported timestamp map back to the
original uploaded file.

Real pixel decoding is attempted via OpenCV when available (needed once real
detector adapters are swapped in); if OpenCV is not installed, frames are
still produced with correct timestamps and `array=None`, which is sufficient
for the current mock detector adapters that only use `frame_ms`.
"""

from __future__ import annotations

from pathlib import Path

from yoyovision_ml.interfaces import FrameRef


def sample_frame_timestamps_ms(duration_ms: int, fps: float, sample_fps: float) -> list[int]:
    """Deterministically computes the original-timeline millisecond offsets
    to sample at, given the source video's duration/fps and a target sampling
    rate (which may be lower than the source fps to bound compute cost)."""
    if duration_ms <= 0 or fps <= 0 or sample_fps <= 0:
        return []
    effective_sample_fps = min(sample_fps, fps)
    step_ms = 1000.0 / effective_sample_fps
    timestamps: list[int] = []
    t = 0.0
    while t < duration_ms:
        timestamps.append(int(round(t)))
        t += step_ms
    return timestamps


def extract_frames(
    video_path: Path, duration_ms: int, fps: float, sample_fps: float = 15.0
) -> list[FrameRef]:
    """Returns `FrameRef`s at `sample_fps`, each stamped with its true
    original-video millisecond offset."""
    timestamps = sample_frame_timestamps_ms(duration_ms, fps, sample_fps)
    if not timestamps:
        return []

    try:
        import cv2
    except ImportError:
        return [FrameRef(frame_ms=ts, array=None) for ts in timestamps]

    frames: list[FrameRef] = []
    capture = cv2.VideoCapture(str(video_path))
    try:
        for ts in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(ts))
            ok, frame = capture.read()
            frames.append(FrameRef(frame_ms=ts, array=frame if ok else None))
    finally:
        capture.release()
    return frames
