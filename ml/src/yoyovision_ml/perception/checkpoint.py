"""Yo-yo detector checkpoint I/O."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from yoyovision_ml.perception.config import DetectorTrainingConfig

CHECKPOINT_SCHEMA_VERSION = "yoyo-detector-checkpoint-v1"
DEFAULT_MODEL_NAME = "tiny-yoyo-detector"


class YoyoDetectorMetadata(BaseModel):
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    model_name: str
    model_version: str
    training_config: DetectorTrainingConfig
    input_size: int = 128
    player_splits: dict[str, str]
    best_epoch: int
    val_loss_history: list[float]
    val_metrics: dict[str, object]
    test_metrics: dict[str, object] | None
    train_sample_count: int
    val_sample_count: int
    test_sample_count: int
    torch_version: str
    training_data_source: str = "synthetic"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def save_checkpoint(
    torch_module: Any,
    model: Any,
    metadata: YoyoDetectorMetadata,
    output_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / f"{name}.pt"
    metadata_path = output_dir / f"{name}.json"
    torch_module.save(model.state_dict(), weights_path)
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return weights_path, metadata_path


def load_checkpoint(
    torch_module: Any, weights_path: Path
) -> tuple[dict[str, Any], YoyoDetectorMetadata]:
    metadata_path = weights_path.with_suffix(".json")
    metadata = YoyoDetectorMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    state_dict = torch_module.load(weights_path, map_location="cpu", weights_only=True)
    return state_dict, metadata
