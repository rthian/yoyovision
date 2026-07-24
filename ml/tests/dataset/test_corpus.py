from __future__ import annotations

import json
from pathlib import Path

from yoyovision_ml.dataset.corpus import append_record_to_corpus, ensure_corpus_dir
from yoyovision_ml.dataset.io import load_dataset, load_record
from yoyovision_ml.dataset.schema import DatasetRecord, DatasetVideo


def _video(video_id: str = "video-1") -> DatasetVideo:
    return DatasetVideo(
        video_id=video_id,
        player_id="player-1",
        division="1A",
        relative_path="uploads/video-1.mp4",
        checksum_sha256="b" * 64,
        duration_ms=12_000,
        width=1280,
        height=720,
        source_fps=30.0,
    )


def _record(record_id: str = "analysis-1__dev_at_yoyovision.local") -> DatasetRecord:
    return DatasetRecord(
        record_id=record_id,
        video=_video(),
        annotator_id="dev@yoyovision.local",
        is_adjudicated=True,
        schema_version="1.0.0",
        ontology_version="dataset-ontology-v1",
        trick_events=[],
        deductions=[],
        freestyle_evaluations=[],
    )


def test_ensure_corpus_dir_creates_manifest_and_subdirs(tmp_path: Path) -> None:
    manifest = ensure_corpus_dir(tmp_path)
    assert manifest.dataset_version == "corpus-v1"
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "videos").is_dir()
    assert (tmp_path / "records").is_dir()


def test_append_record_to_corpus_writes_video_record_and_manifest(tmp_path: Path) -> None:
    record = _record()
    video_bytes = b"fake-mp4-bytes"

    record_path = append_record_to_corpus(
        tmp_path,
        record,
        video_bytes,
        video_filename="routine.mp4",
    )

    assert record_path.exists()
    assert (tmp_path / "videos/video-1.mp4").read_bytes() == video_bytes

    loaded = load_record(record_path)
    assert loaded.video.relative_path == "videos/video-1.mp4"
    assert loaded.video.checksum_sha256 != "b" * 64

    manifest, records = load_dataset(tmp_path)
    assert manifest.video_ids == ["video-1"]
    assert len(records) == 1
    assert records[0].record_id == record.record_id


def test_append_record_to_corpus_replaces_existing_record(tmp_path: Path) -> None:
    record = _record()
    first_path = append_record_to_corpus(tmp_path, record, b"v1", video_filename="a.mp4")
    second_path = append_record_to_corpus(tmp_path, record, b"v2", video_filename="a.mp4")

    assert first_path == second_path
    assert (tmp_path / "videos/video-1.mp4").read_bytes() == b"v2"

    manifest, records = load_dataset(tmp_path)
    assert len(manifest.record_paths) == 1
    assert len(records) == 1
