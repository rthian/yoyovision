"""Append reviewed `DatasetRecord` files into a versioned dataset directory."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from yoyovision_ml.dataset.io import load_manifest, save_manifest, save_record
from yoyovision_ml.dataset.ontology import default_ontology
from yoyovision_ml.dataset.schema import DatasetManifest, DatasetRecord, DatasetVideo


class CorpusError(Exception):
    """Raised when a record cannot be appended to the training corpus."""


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_corpus_dir(corpus_dir: Path) -> DatasetManifest:
    """Creates `corpus_dir` and an empty manifest when missing."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "videos").mkdir(parents=True, exist_ok=True)
    (corpus_dir / "records").mkdir(parents=True, exist_ok=True)
    manifest_path = corpus_dir / "manifest.json"
    if manifest_path.exists():
        return load_manifest(corpus_dir)
    manifest = DatasetManifest(
        dataset_version="corpus-v1",
        ontology_version=default_ontology().version,
        created_at=datetime.now(UTC),
        video_ids=[],
        record_paths=[],
        notes="YoYoVision training corpus exported from submitted analysis reviews.",
    )
    save_manifest(corpus_dir, manifest)
    return manifest


def _video_extension(video_filename: str | None, fallback: str = ".mp4") -> str:
    if not video_filename:
        return fallback
    suffix = Path(video_filename).suffix
    return suffix if suffix else fallback


def append_record_to_corpus(
    corpus_dir: Path,
    record: DatasetRecord,
    video_bytes: bytes,
    *,
    video_filename: str | None = None,
) -> Path:
    """Writes the video file, record JSON, and updates `manifest.json`.

    Re-appending the same `record_id` replaces the prior record entry and
    refreshes the on-disk video bytes.
    """
    manifest = ensure_corpus_dir(corpus_dir)
    extension = _video_extension(video_filename)
    relative_video_path = f"videos/{record.video.video_id}{extension}"
    video_path = corpus_dir / relative_video_path
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(video_bytes)

    checksum = _sha256_hex(video_bytes)
    corpus_video = DatasetVideo(
        **record.video.model_dump()
        | {
            "relative_path": relative_video_path,
            "checksum_sha256": checksum,
        }
    )
    corpus_record = record.model_copy(update={"video": corpus_video})
    record_path = save_record(corpus_dir, corpus_record)
    relative_record_path = str(record_path.relative_to(corpus_dir))

    video_ids = [vid for vid in manifest.video_ids if vid != corpus_video.video_id]
    video_ids.append(corpus_video.video_id)

    record_paths = [
        path for path in manifest.record_paths if not path.endswith(f"/{corpus_record.record_id}.json")
    ]
    record_paths = [path for path in record_paths if not path.endswith(f"{corpus_record.record_id}.json")]
    if relative_record_path not in record_paths:
        record_paths.append(relative_record_path)

    updated_manifest = manifest.model_copy(
        update={
            "video_ids": video_ids,
            "record_paths": record_paths,
        }
    )
    save_manifest(corpus_dir, updated_manifest)
    return record_path
