"""Prediction artefact I/O: a Parquet event-prediction table + JSON metadata
sidecar, per Prompt C's inference contract, mirroring `perception.artifact`'s
Parquet-plus-JSON-sidecar convention on the feature side.

One row per predicted `AnalysisEventPrediction`; `evidence` is stored as a
JSON string column (a list of `{frame_ms, bbox, keypoint_refs, note}`
dicts) since it does not fit tabular columns cleanly and could in principle
contain more than one entry, even though every adapter in this package
today (`events.convert.to_analysis_event_prediction`) always emits exactly
one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from yoyovision_ml.domain import (
    AnalysisEventPrediction,
    BoundingBox,
    DifficultyBand,
    EventFamily,
    EvidenceRef,
    Outcome,
)

ARTIFACT_SCHEMA_VERSION = "events-prediction-artifact-v1"

_COLUMNS = (
    "label",
    "family",
    "start_ms",
    "end_ms",
    "confidence",
    "outcome",
    "difficulty_band",
    "model_name",
    "model_version",
    "evidence",
)


class PredictionArtifactMetadata(BaseModel):
    """JSON sidecar written next to the Parquet prediction table."""

    schema_version: str = ARTIFACT_SCHEMA_VERSION
    video_id: str
    model_name: str
    model_version: str
    event_count: int = Field(ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _evidence_to_list(evidence: tuple[EvidenceRef, ...]) -> list[dict[str, Any]]:
    return [
        {
            "frame_ms": ref.frame_ms,
            "bbox": (
                {
                    "x": ref.bbox.x,
                    "y": ref.bbox.y,
                    "width": ref.bbox.width,
                    "height": ref.bbox.height,
                }
                if ref.bbox is not None
                else None
            ),
            "keypoint_refs": list(ref.keypoint_refs),
            "note": ref.note,
        }
        for ref in evidence
    ]


def _evidence_from_list(raw: list[dict[str, Any]]) -> tuple[EvidenceRef, ...]:
    refs: list[EvidenceRef] = []
    for item in raw:
        bbox_data = item.get("bbox")
        bbox = BoundingBox(**bbox_data) if bbox_data else None
        refs.append(
            EvidenceRef(
                frame_ms=int(item["frame_ms"]),
                bbox=bbox,
                keypoint_refs=tuple(item.get("keypoint_refs", [])),
                note=str(item.get("note", "")),
            )
        )
    return tuple(refs)


def write_predictions(
    predictions: list[AnalysisEventPrediction],
    video_id: str,
    output_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    """Writes `<output_dir>/<name>.parquet` + `<output_dir>/<name>.json`.

    `model_name`/`model_version` in the metadata sidecar are read from the
    first prediction -- every prediction from one `TemporalEventDetector.predict()`
    call shares the same model identity. An empty `predictions` list still
    writes a valid (empty) artefact, with `model_name`/`model_version` left
    as `"unknown"` since there is nothing to read them from in that case.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{name}.parquet"
    metadata_path = output_dir / f"{name}.json"

    rows = [
        {
            "label": prediction.label,
            "family": prediction.family.value,
            "start_ms": prediction.start_ms,
            "end_ms": prediction.end_ms,
            "confidence": prediction.confidence,
            "outcome": str(prediction.outcome),
            "difficulty_band": str(prediction.difficulty_band),
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
            "evidence": json.dumps(_evidence_to_list(prediction.evidence)),
        }
        for prediction in predictions
    ]
    frame = pd.DataFrame(rows, columns=list(_COLUMNS))
    frame.to_parquet(parquet_path, index=False)

    metadata = PredictionArtifactMetadata(
        video_id=video_id,
        model_name=predictions[0].model_name if predictions else "unknown",
        model_version=predictions[0].model_version if predictions else "unknown",
        event_count=len(predictions),
    )
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return parquet_path, metadata_path


def read_predictions(
    parquet_path: Path,
) -> tuple[list[AnalysisEventPrediction], PredictionArtifactMetadata]:
    """Inverse of `write_predictions`, given the `.parquet` path (sidecar
    `.json` is located by replacing the suffix)."""
    metadata_path = parquet_path.with_suffix(".json")
    metadata = PredictionArtifactMetadata.model_validate_json(
        metadata_path.read_text(encoding="utf-8")
    )

    df = pd.read_parquet(parquet_path)
    predictions = [
        AnalysisEventPrediction(
            label=str(row["label"]),
            family=EventFamily(row["family"]),
            start_ms=int(row["start_ms"]),
            end_ms=int(row["end_ms"]),
            confidence=float(row["confidence"]),
            outcome=Outcome(row["outcome"]),
            difficulty_band=DifficultyBand(row["difficulty_band"]),
            model_name=str(row["model_name"]),
            model_version=str(row["model_version"]),
            evidence=_evidence_from_list(json.loads(row["evidence"])),
        )
        for _, row in df.iterrows()
    ]
    return predictions, metadata
