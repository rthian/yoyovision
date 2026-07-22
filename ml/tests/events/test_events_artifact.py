from __future__ import annotations

from pathlib import Path

from yoyovision_ml.domain import (
    AnalysisEventPrediction,
    BoundingBox,
    DifficultyBand,
    EventFamily,
    EvidenceRef,
    Outcome,
)
from yoyovision_ml.events.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    read_predictions,
    write_predictions,
)


def _prediction(**overrides: object) -> AnalysisEventPrediction:
    defaults: dict[str, object] = {
        "label": "hop",
        "family": EventFamily.HOP,
        "start_ms": 100,
        "end_ms": 300,
        "confidence": 0.87,
        "outcome": Outcome.SUCCESS,
        "difficulty_band": DifficultyBand.UNKNOWN,
        "model_name": "trick-event-tcn",
        "model_version": "v-1",
        "evidence": (EvidenceRef(frame_ms=100, note="test evidence"),),
    }
    defaults.update(overrides)
    return AnalysisEventPrediction(**defaults)  # type: ignore[arg-type]


def test_write_predictions_creates_a_parquet_file_and_a_json_sidecar(tmp_path: Path) -> None:
    parquet_path, metadata_path = write_predictions(
        [_prediction()], "video-1", tmp_path, "predictions"
    )
    assert parquet_path == tmp_path / "predictions.parquet"
    assert metadata_path == tmp_path / "predictions.json"
    assert parquet_path.exists()
    assert metadata_path.exists()


def test_read_predictions_round_trips_every_field(tmp_path: Path) -> None:
    bbox = BoundingBox(x=1.0, y=2.0, width=3.0, height=4.0)
    original = _prediction(
        label="whip_catch",
        family=EventFamily.WHIP_CATCH,
        start_ms=500,
        end_ms=750,
        confidence=0.62,
        outcome=Outcome.MISS,
        evidence=(EvidenceRef(frame_ms=500, bbox=bbox, keypoint_refs=("wrist_l",), note="n"),),
    )
    parquet_path, _ = write_predictions([original], "video-2", tmp_path, "predictions")

    predictions, metadata = read_predictions(parquet_path)

    assert len(predictions) == 1
    restored = predictions[0]
    assert restored.label == "whip_catch"
    assert restored.family == EventFamily.WHIP_CATCH
    assert restored.start_ms == 500
    assert restored.end_ms == 750
    assert restored.confidence == 0.62
    assert restored.outcome == Outcome.MISS
    assert restored.difficulty_band == DifficultyBand.UNKNOWN
    assert restored.model_name == "trick-event-tcn"
    assert restored.model_version == "v-1"
    assert len(restored.evidence) == 1
    assert restored.evidence[0].frame_ms == 500
    assert restored.evidence[0].bbox == bbox
    assert restored.evidence[0].keypoint_refs == ("wrist_l",)
    assert restored.evidence[0].note == "n"
    assert metadata.schema_version == ARTIFACT_SCHEMA_VERSION


def test_metadata_reads_model_identity_from_the_first_prediction(tmp_path: Path) -> None:
    predictions = [
        _prediction(model_name="model-a", model_version="v-a"),
        _prediction(model_name="model-a", model_version="v-a"),
    ]
    parquet_path, _ = write_predictions(predictions, "video-3", tmp_path, "predictions")

    _, metadata = read_predictions(parquet_path)

    assert metadata.model_name == "model-a"
    assert metadata.model_version == "v-a"
    assert metadata.event_count == 2
    assert metadata.video_id == "video-3"


def test_write_predictions_of_an_empty_list_still_writes_a_valid_artifact(tmp_path: Path) -> None:
    parquet_path, _ = write_predictions([], "video-empty", tmp_path, "predictions")

    predictions, metadata = read_predictions(parquet_path)

    assert predictions == []
    assert metadata.model_name == "unknown"
    assert metadata.model_version == "unknown"
    assert metadata.event_count == 0


def test_evidence_without_a_bbox_round_trips_as_none(tmp_path: Path) -> None:
    original = _prediction(evidence=(EvidenceRef(frame_ms=42, bbox=None, note="no bbox"),))
    parquet_path, _ = write_predictions([original], "video-4", tmp_path, "predictions")

    predictions, _ = read_predictions(parquet_path)

    assert predictions[0].evidence[0].bbox is None
