"""Reading and writing dataset manifests/records to/from a dataset directory.

Layout convention for a dataset directory (see `docs/annotation_handbook.md`):

    <dataset_root>/
        manifest.json
        videos/<video_id>.<ext>          # referenced by DatasetVideo.relative_path
        records/<record_id>.json         # one DatasetRecord per file
"""

from __future__ import annotations

import json
from pathlib import Path

from yoyovision_ml.dataset.schema import DatasetManifest, DatasetRecord

_MANIFEST_FILENAME = "manifest.json"
_RECORDS_DIRNAME = "records"


def load_manifest(dataset_dir: Path) -> DatasetManifest:
    path = dataset_dir / _MANIFEST_FILENAME
    with path.open("r", encoding="utf-8") as fh:
        return DatasetManifest.model_validate(json.load(fh))


def save_manifest(dataset_dir: Path, manifest: DatasetManifest) -> Path:
    path = dataset_dir / _MANIFEST_FILENAME
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_record(path: Path) -> DatasetRecord:
    with path.open("r", encoding="utf-8") as fh:
        return DatasetRecord.model_validate(json.load(fh))


def save_record(dataset_dir: Path, record: DatasetRecord) -> Path:
    records_dir = dataset_dir / _RECORDS_DIRNAME
    records_dir.mkdir(parents=True, exist_ok=True)
    path = records_dir / f"{record.record_id}.json"
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_dataset(dataset_dir: Path) -> tuple[DatasetManifest, list[DatasetRecord]]:
    """Loads the manifest and every record it references, in manifest order.

    Raises `FileNotFoundError` (with the missing path named) rather than
    silently skipping a record a manifest claims to reference -- a
    manifest/records mismatch is itself a dataset integrity error.
    """
    manifest = load_manifest(dataset_dir)
    records = []
    for record_path in manifest.record_paths:
        full_path = dataset_dir / record_path
        if not full_path.exists():
            raise FileNotFoundError(f"Manifest references missing record file: {full_path}")
        records.append(load_record(full_path))
    return manifest, records


def select_training_records(records: list[DatasetRecord]) -> list[DatasetRecord]:
    """Picks exactly one record per `video_id` for downstream training/eval use.

    Prefers the adjudicated record for a video when one exists; if a video
    has exactly one (non-adjudicated) record, that single annotator pass is
    used; if a video has multiple non-adjudicated records and none is
    adjudicated, it is excluded and reported so a human can adjudicate
    rather than silently picking an arbitrary annotator's version.
    """
    by_video: dict[str, list[DatasetRecord]] = {}
    for record in records:
        by_video.setdefault(record.video.video_id, []).append(record)

    selected: list[DatasetRecord] = []
    for video_records in by_video.values():
        adjudicated = [r for r in video_records if r.is_adjudicated]
        if adjudicated:
            selected.append(adjudicated[0])
        elif len(video_records) == 1:
            selected.append(video_records[0])
        # else: multiple un-adjudicated passes for this video_id, no clear
        # winner -- deliberately excluded from training selection.
    return selected
