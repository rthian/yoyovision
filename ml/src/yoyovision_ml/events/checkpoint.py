"""Trained-model checkpoint I/O: `.pt` weights + JSON metadata sidecar.

Mirrors `perception.artifact`'s Parquet-plus-JSON-sidecar convention, but for
a `torch` `state_dict` (`torch.save`/`torch.load` handle the binary side)
alongside a plain Pydantic `EventModelMetadata` JSON sidecar carrying
everything needed to *reconstruct* inference without retraining: the
`TrainingConfig` used (Prompt C: "experiment configuration saved with each
run"), feature-column order, normalization stats, per-class calibration
temperatures, the player-grouped split assignment, and headline
validation/test metrics.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from yoyovision_ml.events.config import TrainingConfig

CHECKPOINT_SCHEMA_VERSION = "events-checkpoint-v1"


class EventModelMetadata(BaseModel):
    """JSON sidecar written next to the `.pt` weights file."""

    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    model_name: str
    model_version: str
    training_config: TrainingConfig
    feature_names: tuple[str, ...]
    input_dim: int
    normalization: dict[str, list[float]]
    #: One calibrated temperature per `labels.EVENT_CLASSES`-order class.
    calibration_temperatures: list[float]
    #: `player_id -> "train"|"val"|"test"`, for leakage auditing after the fact.
    player_splits: dict[str, str]
    best_epoch: int
    val_loss_history: list[float]
    val_metrics: dict[str, object]
    test_metrics: dict[str, object] | None
    train_sample_count: int
    val_sample_count: int
    test_sample_count: int
    torch_version: str
    #: `"synthetic"` (no real footage -- see `events/synthetic.py`) or
    #: `"dataset"` (real Prompt A/B annotated clips). Never omit this --
    #: Prompt C: "Do not claim production readiness based only on
    #: clip-level accuracy."
    training_data_source: str = "synthetic"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def save_checkpoint(
    torch_module: Any,
    model: Any,
    metadata: EventModelMetadata,
    output_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    """Writes `<output_dir>/<name>.pt` + `<output_dir>/<name>.json`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / f"{name}.pt"
    metadata_path = output_dir / f"{name}.json"
    torch_module.save(model.state_dict(), weights_path)
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return weights_path, metadata_path


def load_checkpoint(
    torch_module: Any, weights_path: Path
) -> tuple[dict[str, Any], EventModelMetadata]:
    """Inverse of `save_checkpoint`, given the `.pt` path (sidecar `.json` is
    located by replacing the suffix). Returns `(state_dict, metadata)` --
    callers build the model architecture themselves (via
    `model.build_model`, using `metadata.training_config`/`metadata.input_dim`)
    and then `model.load_state_dict(state_dict)`.

    `weights_only=True` restricts unpickling to tensors/plain Python
    containers -- a checkpoint file is untrusted input, same rationale as
    `perception/detector_pytorch.py`."""
    metadata_path = weights_path.with_suffix(".json")
    metadata = EventModelMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    state_dict = torch_module.load(weights_path, map_location="cpu", weights_only=True)
    return state_dict, metadata
