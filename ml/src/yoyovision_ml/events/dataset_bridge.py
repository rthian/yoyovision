"""Bridge Prompt A `DatasetRecord` annotations to Prompt C `TrainingSample`s.

Pairs adjudicated (or single-pass) dataset records with Prompt B perception
Parquet artefacts produced by `yoyovision-perception run`. This is the missing
link between real-footage annotation and `yoyovision-events train`.
"""

from __future__ import annotations

from pathlib import Path

from yoyovision_ml.dataset.io import load_dataset, select_training_records
from yoyovision_ml.dataset.schema import DatasetRecord
from yoyovision_ml.events.types import TrainingSample
from yoyovision_ml.perception.artifact import read_artifact


class DatasetBridgeError(Exception):
    """Raised when a dataset record cannot be paired with perception features."""


def resolve_perception_parquet(perception_dir: Path, video_id: str) -> Path:
    """Locates `<perception_dir>/<video_id>.parquet` (or nested variant)."""
    candidates = [
        perception_dir / f"{video_id}.parquet",
        perception_dir / video_id / f"{video_id}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise DatasetBridgeError(
        f"No perception artefact for video_id={video_id!r} under {perception_dir}. "
        f"Expected one of: {[str(p) for p in candidates]}"
    )


def training_sample_from_record(
    record: DatasetRecord, perception_dir: Path
) -> TrainingSample:
    parquet_path = resolve_perception_parquet(perception_dir, record.video.video_id)
    features, _metadata = read_artifact(parquet_path)
    return TrainingSample(
        video_id=record.video.video_id,
        player_id=record.video.player_id,
        features=features,
        trick_events=tuple(record.trick_events),
    )


def load_training_samples_from_dataset(
    dataset_dir: Path,
    perception_dir: Path,
    *,
    record_ids: set[str] | None = None,
) -> list[TrainingSample]:
    """Loads one `TrainingSample` per training-selected video in `dataset_dir`."""
    _manifest, records = load_dataset(dataset_dir)
    selected = select_training_records(records)
    if record_ids is not None:
        selected = [record for record in selected if record.record_id in record_ids]

    if not selected:
        raise DatasetBridgeError(
            f"No training-eligible records found in {dataset_dir}. "
            "Ensure adjudicated passes exist or each video has a single annotator record."
        )

    samples: list[TrainingSample] = []
    for record in selected:
        samples.append(training_sample_from_record(record, perception_dir))
    return samples
