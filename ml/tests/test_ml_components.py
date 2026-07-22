from __future__ import annotations

from pathlib import Path

import pytest

from yoyovision_ml.adapters_mock import (
    MockPoseEstimator,
    MockTemporalEventDetector,
    MockTracker,
    MockYoyoDetector,
)
from yoyovision_ml.domain import BoundingBox, Detection, FeatureFrame, FeatureSet
from yoyovision_ml.exports import sanitize_export_filename
from yoyovision_ml.interfaces import FrameRef
from yoyovision_ml.media_validation import (
    MediaValidationError,
    sniff_container_mime_type,
    validate_signature,
    validate_size,
)

_MP4_HEADER = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8
_WEBM_HEADER = b"\x1a\x45\xdf\xa3" + b"\x00" * 16


def test_sniff_mp4_signature() -> None:
    assert sniff_container_mime_type(_MP4_HEADER) == "video/mp4"


def test_sniff_webm_signature() -> None:
    assert sniff_container_mime_type(_WEBM_HEADER) == "video/webm"


def test_sniff_unrecognized_returns_none() -> None:
    assert sniff_container_mime_type(b"not a video") is None


def test_validate_signature_mismatch_raises() -> None:
    with pytest.raises(MediaValidationError) as exc_info:
        validate_signature(_WEBM_HEADER, "video/mp4")
    assert exc_info.value.code == "signature_mismatch"


def test_validate_signature_disallowed_mime_raises() -> None:
    with pytest.raises(MediaValidationError) as exc_info:
        validate_signature(_MP4_HEADER, "application/octet-stream")
    assert exc_info.value.code == "unsupported_mime_type"


def test_validate_size_rejects_oversized_file() -> None:
    with pytest.raises(MediaValidationError) as exc_info:
        validate_size(1_000_000, max_bytes=500_000)
    assert exc_info.value.code == "file_too_large"


def test_validate_size_rejects_empty_file() -> None:
    with pytest.raises(MediaValidationError) as exc_info:
        validate_size(0, max_bytes=500_000)
    assert exc_info.value.code == "empty_file"


def test_sanitize_export_filename_strips_path_traversal() -> None:
    result = sanitize_export_filename("../../etc/passwd", "csv")
    assert "/" not in result
    assert ".." not in result
    assert result.endswith(".csv")


def test_sanitize_export_filename_ignores_client_extension() -> None:
    result = sanitize_export_filename("report.exe", "json")
    assert result.endswith(".json")
    assert "exe" not in result


def test_mock_pose_estimator_is_deterministic() -> None:
    estimator = MockPoseEstimator()
    path = Path("/tmp/example-video.mp4")
    first = estimator.predict(path)
    second = estimator.predict(path)
    assert first.model_name.startswith("mock-")
    assert [f.keypoints for f in first.frames] == [f.keypoints for f in second.frames]


def test_mock_pose_estimator_uses_probed_video_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    from yoyovision_ml import adapters_mock

    path = Path("/tmp/long-routine.mp4")
    monkeypatch.setattr(
        adapters_mock,
        "_mock_video_timeline",
        lambda _video_path: (30.0, 221_860),
    )
    sequence = MockPoseEstimator().predict(path)
    assert sequence.frames[-1].frame_ms >= 221_000


def test_mock_yoyo_detector_is_deterministic_and_labelled() -> None:
    detector = MockYoyoDetector()
    frames = [FrameRef(frame_ms=0, array=None), FrameRef(frame_ms=33, array=None)]
    first = detector.predict(frames)
    second = detector.predict(frames)
    assert detector.model_name.startswith("mock-")
    assert [d.bbox for d in first] == [d.bbox for d in second]


def test_mock_tracker_passes_through_detection_at_matching_timestamp() -> None:
    tracker = MockTracker()
    detection = Detection(
        frame_ms=100,
        bbox=BoundingBox(x=0.1, y=0.1, width=0.05, height=0.05),
        confidence=0.8,
        class_label="yoyo",
        model_name="mock-yoyo-detector",
        model_version="0.0.0-mock",
    )
    tracks = tracker.update([detection], timestamp_ms=100)
    assert len(tracks) == 1
    assert tracks[0].track_id == "track-0"


def test_mock_temporal_event_detector_spans_full_duration() -> None:
    detector = MockTemporalEventDetector()
    features = FeatureSet(
        frames=tuple(FeatureFrame(frame_ms=ms, values={}) for ms in range(0, 10_000, 100)),
        feature_names=(),
        fps=10.0,
    )
    events, deductions = detector.predict(features)
    assert len(events) > 0
    assert all(e.model_name == "mock-temporal-event-detector" for e in events)
    assert all(e.end_ms <= features.frames[-1].frame_ms for e in events)


def test_mock_temporal_event_detector_empty_features_returns_empty() -> None:
    detector = MockTemporalEventDetector()
    events, deductions = detector.predict(FeatureSet(frames=(), feature_names=(), fps=0.0))
    assert events == []
    assert deductions == []
