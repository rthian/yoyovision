from __future__ import annotations

from pathlib import Path

from yoyovision_ml.perception.artifact import read_artifact
from yoyovision_ml.perception.pipeline import PerceptionPipeline


def test_run_with_mock_adapters_produces_nonempty_feature_set(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"not a real video, mock adapters ignore pixel content")

    pipeline = PerceptionPipeline(sample_fps=15.0)
    result = pipeline.run(video_path, duration_ms=2000, fps=30.0)

    assert len(result.feature_set.frames) > 0
    assert len(result.yoyo_tracks) > 0
    assert len(result.yoyo_detections) > 0
    assert result.pose_sequence.model_name == "mock-pose-estimator"
    assert result.hand_sequence.model_name == "mock-hand-estimator"


def test_run_is_deterministic_for_same_video_path(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"same content")

    pipeline_a = PerceptionPipeline(sample_fps=15.0)
    pipeline_b = PerceptionPipeline(sample_fps=15.0)
    result_a = pipeline_a.run(video_path, duration_ms=1000, fps=30.0)
    result_b = pipeline_b.run(video_path, duration_ms=1000, fps=30.0)

    assert [f.frame_ms for f in result_a.feature_set.frames] == [
        f.frame_ms for f in result_b.feature_set.frames
    ]
    assert result_a.feature_set.frames[0].values == result_b.feature_set.frames[0].values


def test_metadata_reports_model_versions_for_every_adapter(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"content")

    pipeline = PerceptionPipeline(sample_fps=15.0)
    result = pipeline.run(video_path, duration_ms=1000, fps=30.0)

    assert set(result.metadata.model_versions) == {
        "pose_estimator",
        "hand_estimator",
        "yoyo_detector",
        "tracker",
    }
    assert result.metadata.frame_count == len(result.feature_set.frames)
    # MockTracker has no `track_quality` method (only the Kalman tracker
    # does), so the pipeline should leave this as None rather than error.
    assert result.metadata.track_quality is None


def test_metadata_video_checksum_is_unavailable_for_nonexistent_video(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.mp4"
    pipeline = PerceptionPipeline(sample_fps=15.0)
    result = pipeline.run(missing_path, duration_ms=1000, fps=30.0)
    assert result.metadata.video_checksum_sha256 == "unavailable"


def test_metadata_video_checksum_is_computed_for_existing_video(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"real bytes for checksum")
    pipeline = PerceptionPipeline(sample_fps=15.0)
    result = pipeline.run(video_path, duration_ms=1000, fps=30.0)
    assert result.metadata.video_checksum_sha256 != "unavailable"
    assert len(result.metadata.video_checksum_sha256) == 64


def test_run_and_write_produces_readable_artifact(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"content")
    output_dir = tmp_path / "artifacts"

    pipeline = PerceptionPipeline(sample_fps=15.0)
    result, parquet_path, metadata_path = pipeline.run_and_write(
        video_path, duration_ms=1000, fps=30.0, output_dir=output_dir, name="clip"
    )

    assert parquet_path.exists()
    assert metadata_path.exists()
    feature_set, metadata = read_artifact(parquet_path)
    assert len(feature_set.frames) == len(result.feature_set.frames)
    assert metadata.video_filename == "video.mp4"


def test_kalman_tracker_adapter_kwargs_are_forwarded(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"content")

    pipeline = PerceptionPipeline(
        tracker_adapter_name="kalman",
        tracker_adapter_kwargs={"max_gap_ms": 100, "static_camera": True},
        sample_fps=15.0,
    )
    result = pipeline.run(video_path, duration_ms=1000, fps=30.0)

    assert result.metadata.model_versions["tracker"].startswith("kalman-yoyo-tracker@")
    assert result.metadata.track_quality is not None
