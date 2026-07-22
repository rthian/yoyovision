from __future__ import annotations

from pathlib import Path

import pytest

from yoyovision_ml.domain import FeatureFrame, FeatureSet
from yoyovision_ml.events.dataset_bridge import (
    DatasetBridgeError,
    load_training_samples_from_dataset,
    resolve_perception_parquet,
)
from yoyovision_ml.perception.artifact import PerceptionMetadata, write_artifact
from yoyovision_ml.perception.features import ALL_FEATURE_NAMES


def _write_dummy_perception(perception_dir: Path, video_id: str) -> None:
    feature_set = FeatureSet(
        frames=(
            FeatureFrame(frame_ms=0, values={ALL_FEATURE_NAMES[0]: 0.5}),
            FeatureFrame(frame_ms=1000, values={ALL_FEATURE_NAMES[0]: 0.6}),
        ),
        feature_names=ALL_FEATURE_NAMES,
        fps=15.0,
    )
    metadata = PerceptionMetadata(
        video_filename=f"{video_id}.mp4",
        video_checksum_sha256="abc",
        duration_ms=20_000,
        source_fps=30.0,
        processed_fps=15.0,
        frame_count=2,
        preprocessing_version="test",
        model_versions={"pose": "mock@0"},
        feature_names=ALL_FEATURE_NAMES,
    )
    write_artifact(feature_set, metadata, perception_dir, video_id)


def test_resolve_perception_parquet_finds_video_keyed_file(tmp_path: Path) -> None:
    parquet = tmp_path / "sample_video_001.parquet"
    parquet.write_bytes(b"not-real")
    assert resolve_perception_parquet(tmp_path, "sample_video_001") == parquet


def test_load_training_samples_from_sample_dataset(tmp_path: Path) -> None:
    dataset_root = Path("ml/sample_data/dataset_v1")
    perception_dir = tmp_path / "perception"
    perception_dir.mkdir()
    for video_id in ("sample_video_001", "sample_video_002", "sample_video_003"):
        _write_dummy_perception(perception_dir, video_id)

    samples = load_training_samples_from_dataset(dataset_root, perception_dir)
    assert len(samples) == 3
    assert {sample.video_id for sample in samples} == {
        "sample_video_001",
        "sample_video_002",
        "sample_video_003",
    }
    assert all(sample.features.feature_names == ALL_FEATURE_NAMES for sample in samples)


def test_load_training_samples_raises_when_perception_missing(tmp_path: Path) -> None:
    dataset_root = Path("ml/sample_data/dataset_v1")
    with pytest.raises(DatasetBridgeError):
        load_training_samples_from_dataset(dataset_root, tmp_path / "empty")
