"""Perception feature artefact I/O: a Parquet feature table + JSON metadata
sidecar, per Prompt B's "Write a timestamped Parquet or NPZ feature artefact
plus JSON metadata."

Parquet (via pandas + the now-core `pyarrow` dependency) was chosen over NPZ
for the tabular per-frame feature table because pandas is already a core
`ml` dependency and Parquet keeps column names alongside the data, whereas an
`.npz` would need the JSON sidecar to carry column ordering too. Missing
values in the table use `NaN` (see `MISSING_VALUE_REPRESENTATION` below),
which Parquet/pandas represent natively -- never a sentinel like `-1` that
could be confused with a real feature value.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from yoyovision_ml.domain import FeatureFrame, FeatureSet

#: Documented once here, referenced from every artifact's metadata sidecar.
COORDINATE_CONVENTION = "normalized_0_1_top_left_origin_x_right_y_down"
MISSING_VALUE_REPRESENTATION = "NaN (pandas/numpy float NaN; absent feature for that frame)"
ARTIFACT_SCHEMA_VERSION = "perception-artifact-v1"


class PerceptionMetadata(BaseModel):
    """JSON sidecar written next to the Parquet feature table."""

    schema_version: str = ARTIFACT_SCHEMA_VERSION
    video_filename: str = Field(
        description="Basename only -- never a local absolute path (no path disclosure)."
    )
    video_checksum_sha256: str
    duration_ms: int = Field(ge=0)
    source_fps: float = Field(gt=0)
    processed_fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    coordinate_convention: str = COORDINATE_CONVENTION
    missing_value_representation: str = MISSING_VALUE_REPRESENTATION
    preprocessing_version: str
    model_versions: dict[str, str]
    feature_names: tuple[str, ...]
    track_quality: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def compute_video_checksum(video_path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 -- never loads the whole (potentially large) video into memory."""
    digest = hashlib.sha256()
    with video_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact(
    feature_set: FeatureSet, metadata: PerceptionMetadata, output_dir: Path, name: str
) -> tuple[Path, Path]:
    """Writes `<output_dir>/<name>.parquet` + `<output_dir>/<name>.json`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{name}.parquet"
    metadata_path = output_dir / f"{name}.json"

    rows = [
        {"frame_ms": frame.frame_ms, **{n: frame.values.get(n) for n in feature_set.feature_names}}
        for frame in feature_set.frames
    ]
    columns = ["frame_ms", *feature_set.feature_names]
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_parquet(parquet_path, index=False)

    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return parquet_path, metadata_path


def read_artifact(parquet_path: Path) -> tuple[FeatureSet, PerceptionMetadata]:
    """Inverse of `write_artifact`, given the `.parquet` path (sidecar `.json`
    is located by replacing the suffix)."""
    metadata_path = parquet_path.with_suffix(".json")
    metadata = PerceptionMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))

    df = pd.read_parquet(parquet_path)
    feature_names = metadata.feature_names
    frames = tuple(
        FeatureFrame(
            frame_ms=int(row["frame_ms"]),
            values={name: float(row[name]) for name in feature_names if pd.notna(row[name])},
        )
        for _, row in df.iterrows()
    )
    feature_set = FeatureSet(frames=frames, feature_names=feature_names, fps=metadata.processed_fps)
    return feature_set, metadata


def metadata_to_dict(metadata: PerceptionMetadata) -> dict[str, object]:
    """JSON-safe dict, e.g. for embedding in a larger report (`json.dumps`-ready)."""
    parsed: dict[str, object] = json.loads(metadata.model_dump_json())
    return parsed
