from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from yoyovision_ml.domain import FeatureFrame, FeatureSet
from yoyovision_ml.perception.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    COORDINATE_CONVENTION,
    PerceptionMetadata,
    compute_video_checksum,
    metadata_to_dict,
    read_artifact,
    write_artifact,
)

_FEATURE_NAMES = ("yoyo_x", "yoyo_y")


def _feature_set() -> FeatureSet:
    frames = (
        FeatureFrame(frame_ms=0, values={"yoyo_x": 0.5, "yoyo_y": 0.4}),
        FeatureFrame(frame_ms=100, values={"yoyo_x": 0.55}),  # yoyo_y intentionally missing
    )
    return FeatureSet(frames=frames, feature_names=_FEATURE_NAMES, fps=15.0)


def _metadata(**overrides: object) -> PerceptionMetadata:
    defaults: dict[str, object] = dict(
        video_filename="clip.mp4",
        video_checksum_sha256="a" * 64,
        duration_ms=1000,
        source_fps=30.0,
        processed_fps=15.0,
        frame_count=2,
        preprocessing_version="0.1.0",
        model_versions={"pose_estimator": "mock-pose-estimator@0.0.0-mock"},
        feature_names=_FEATURE_NAMES,
        track_quality=0.8,
    )
    defaults.update(overrides)
    return PerceptionMetadata(**defaults)  # type: ignore[arg-type]


def test_write_artifact_creates_parquet_and_json_files(tmp_path: Path) -> None:
    parquet_path, metadata_path = write_artifact(_feature_set(), _metadata(), tmp_path, "clip")

    assert parquet_path == tmp_path / "clip.parquet"
    assert metadata_path == tmp_path / "clip.json"
    assert parquet_path.exists()
    assert metadata_path.exists()


def test_read_artifact_round_trips_feature_values(tmp_path: Path) -> None:
    parquet_path, _ = write_artifact(_feature_set(), _metadata(), tmp_path, "clip")

    feature_set, metadata = read_artifact(parquet_path)

    assert [f.frame_ms for f in feature_set.frames] == [0, 100]
    assert feature_set.frames[0].values == {"yoyo_x": 0.5, "yoyo_y": 0.4}
    # The missing yoyo_y at frame_ms=100 must round-trip as *absent*, not 0.0.
    assert "yoyo_y" not in feature_set.frames[1].values
    assert feature_set.frames[1].values["yoyo_x"] == 0.55
    assert metadata.schema_version == ARTIFACT_SCHEMA_VERSION
    assert metadata.track_quality == 0.8


def test_read_artifact_uses_processed_fps_for_feature_set(tmp_path: Path) -> None:
    metadata_in = _metadata(processed_fps=12.5)
    parquet_path, _ = write_artifact(_feature_set(), metadata_in, tmp_path, "clip")
    feature_set, metadata = read_artifact(parquet_path)
    assert feature_set.fps == 12.5
    assert metadata.processed_fps == 12.5


def test_compute_video_checksum_matches_hashlib_reference(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    payload = b"fake video bytes" * 1000
    video_path.write_bytes(payload)

    checksum = compute_video_checksum(video_path)

    assert checksum == hashlib.sha256(payload).hexdigest()


def test_compute_video_checksum_streams_in_chunks_without_loading_whole_file(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video.mp4"
    payload = b"x" * 5000
    video_path.write_bytes(payload)

    checksum = compute_video_checksum(video_path, chunk_size=64)

    assert checksum == hashlib.sha256(payload).hexdigest()


def test_metadata_defaults_document_coordinate_convention_and_schema_version() -> None:
    metadata = _metadata()
    assert metadata.coordinate_convention == COORDINATE_CONVENTION
    assert metadata.schema_version == ARTIFACT_SCHEMA_VERSION
    assert "NaN" in metadata.missing_value_representation


def test_metadata_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError):
        _metadata(duration_ms=-1)


def test_metadata_rejects_non_positive_fps() -> None:
    with pytest.raises(ValidationError):
        _metadata(source_fps=0.0)


def test_metadata_to_dict_is_json_safe() -> None:
    metadata = _metadata()
    as_dict = metadata_to_dict(metadata)

    assert isinstance(as_dict, dict)
    assert as_dict["video_filename"] == "clip.mp4"
    assert as_dict["feature_names"] == list(_FEATURE_NAMES)
    # created_at should have been serialized to a string, not a datetime object.
    assert isinstance(as_dict["created_at"], str)


def test_frame_count_matches_written_row_count_by_convention(tmp_path: Path) -> None:
    feature_set = _feature_set()
    metadata = _metadata(frame_count=len(feature_set.frames))
    parquet_path, _ = write_artifact(feature_set, metadata, tmp_path, "clip")

    read_feature_set, read_metadata = read_artifact(parquet_path)
    assert len(read_feature_set.frames) == read_metadata.frame_count


def test_empty_feature_set_writes_and_reads_back_empty(tmp_path: Path) -> None:
    empty = FeatureSet(frames=(), feature_names=_FEATURE_NAMES, fps=15.0)
    metadata = _metadata(frame_count=0)
    parquet_path, _ = write_artifact(empty, metadata, tmp_path, "empty")

    feature_set, _ = read_artifact(parquet_path)
    assert feature_set.frames == ()


def test_nan_values_are_not_confused_with_real_zero(tmp_path: Path) -> None:
    frames = (FeatureFrame(frame_ms=0, values={"yoyo_x": 0.0}),)
    feature_set = FeatureSet(frames=frames, feature_names=_FEATURE_NAMES, fps=15.0)
    parquet_path, _ = write_artifact(feature_set, _metadata(frame_count=1), tmp_path, "zero")

    read_feature_set, _ = read_artifact(parquet_path)
    # A real 0.0 value must be preserved, while the genuinely-missing yoyo_y
    # must be absent (not coerced to 0.0) -- see test_read_artifact_round_trips.
    assert read_feature_set.frames[0].values["yoyo_x"] == 0.0
    assert "yoyo_y" not in read_feature_set.frames[0].values
    assert not math.isnan(read_feature_set.frames[0].values["yoyo_x"])
