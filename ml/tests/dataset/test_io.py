from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from yoyovision_ml.dataset.io import (
    load_dataset,
    load_manifest,
    load_record,
    save_manifest,
    save_record,
    select_training_records,
)
from yoyovision_ml.dataset.schema import DatasetManifest, DatasetRecord, DatasetVideo

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _video(video_id: str = "v1") -> DatasetVideo:
    return DatasetVideo(
        video_id=video_id,
        player_id="p1",
        relative_path=f"videos/{video_id}.mp4",
        checksum_sha256="a" * 64,
        duration_ms=10_000,
        width=1920,
        height=1080,
        source_fps=30.0,
    )


def test_save_and_load_record_round_trips(tmp_path: Path) -> None:
    video = _video()
    record = DatasetRecord(
        record_id="r1", video=video, annotator_id="alex", ontology_version="dataset-ontology-v1"
    )
    path = save_record(tmp_path, record)
    assert path.exists()
    loaded = load_record(path)
    assert loaded == record


def test_save_and_load_manifest_round_trips(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        dataset_version="v1",
        ontology_version="dataset-ontology-v1",
        created_at=NOW,
        video_ids=["v1"],
        record_paths=["records/r1.json"],
    )
    save_manifest(tmp_path, manifest)
    loaded = load_manifest(tmp_path)
    assert loaded == manifest


def test_load_dataset_raises_for_missing_record(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        dataset_version="v1",
        ontology_version="dataset-ontology-v1",
        created_at=NOW,
        video_ids=["v1"],
        record_paths=["records/does_not_exist.json"],
    )
    save_manifest(tmp_path, manifest)
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path)


def test_load_dataset_returns_manifest_and_records(tmp_path: Path) -> None:
    video = _video()
    record = DatasetRecord(
        record_id="r1", video=video, annotator_id="alex", ontology_version="dataset-ontology-v1"
    )
    record_path = save_record(tmp_path, record)
    manifest = DatasetManifest(
        dataset_version="v1",
        ontology_version="dataset-ontology-v1",
        created_at=NOW,
        video_ids=["v1"],
        record_paths=[str(record_path.relative_to(tmp_path))],
    )
    save_manifest(tmp_path, manifest)

    loaded_manifest, loaded_records = load_dataset(tmp_path)
    assert loaded_manifest == manifest
    assert loaded_records == [record]


def test_select_training_records_prefers_adjudicated() -> None:
    video = _video()
    raw = DatasetRecord(
        record_id="raw", video=video, annotator_id="alex", ontology_version="dataset-ontology-v1"
    )
    adjudicated = DatasetRecord(
        record_id="adj",
        video=video,
        annotator_id="alex",
        is_adjudicated=True,
        ontology_version="dataset-ontology-v1",
    )
    selected = select_training_records([raw, adjudicated])
    assert selected == [adjudicated]


def test_select_training_records_uses_single_unadjudicated_pass() -> None:
    video = _video()
    only_pass = DatasetRecord(
        record_id="only",
        video=video,
        annotator_id="alex",
        ontology_version="dataset-ontology-v1",
    )
    selected = select_training_records([only_pass])
    assert selected == [only_pass]


def test_select_training_records_excludes_unresolved_multi_annotator_video() -> None:
    video = _video()
    pass_a = DatasetRecord(
        record_id="a", video=video, annotator_id="alex", ontology_version="dataset-ontology-v1"
    )
    pass_b = DatasetRecord(
        record_id="b", video=video, annotator_id="bo", ontology_version="dataset-ontology-v1"
    )
    selected = select_training_records([pass_a, pass_b])
    assert selected == []
