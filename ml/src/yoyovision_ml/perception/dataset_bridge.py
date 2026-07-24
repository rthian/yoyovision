"""Load yo-yo detector training frames from annotated dataset records."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from yoyovision_ml.dataset.io import load_dataset, select_training_records
from yoyovision_ml.dataset.schema import DatasetRecord, YoyoFrameAnnotation
from yoyovision_ml.domain import BoundingBox
from yoyovision_ml.perception.config import DetectorTrainingConfig
from yoyovision_ml.perception.labels import target_bbox_from_annotation
from yoyovision_ml.perception.types import DetectorTrainingSample
from yoyovision_ml.preprocessing import extract_frames


class DatasetBridgeError(Exception):
    """Raised when dataset records cannot be converted into detector samples."""


def _nearest_frame_ms(frame_ms: int, candidates: list[int]) -> int | None:
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: abs(candidate - frame_ms))


def _samples_for_record(
    dataset_dir: Path,
    record: DatasetRecord,
    config: DetectorTrainingConfig,
) -> list[DetectorTrainingSample]:
    if not record.yoyo_track:
        return []

    video_path = dataset_dir / record.video.relative_path
    if not video_path.exists():
        raise DatasetBridgeError(f"Missing video for {record.record_id}: {video_path}")

    frames = extract_frames(
        video_path,
        duration_ms=record.video.duration_ms,
        fps=record.video.source_fps,
        sample_fps=config.sample_fps,
    )
    usable = [frame for frame in frames if frame.array is not None]
    if not usable:
        raise DatasetBridgeError(
            f"No decodable frames for {record.record_id}. Install OpenCV to train on real videos."
        )

    sampled_ms = [frame.frame_ms for frame in usable]
    step_ms = 1000.0 / min(config.sample_fps, record.video.source_fps)
    tolerance = config.frame_match_tolerance_ms or int(round(step_ms / 2.0))
    frame_by_ms = {frame.frame_ms: frame for frame in usable}

    samples: list[DetectorTrainingSample] = []
    for annotation in record.yoyo_track:
        nearest_ms = _nearest_frame_ms(annotation.frame_ms, sampled_ms)
        if nearest_ms is None or abs(nearest_ms - annotation.frame_ms) > tolerance:
            continue
        frame = frame_by_ms[nearest_ms]
        assert frame.array is not None
        bbox, visible = _annotation_to_target(annotation, config.point_box_size)
        samples.append(
            DetectorTrainingSample(
                video_id=record.video.video_id,
                player_id=record.video.player_id,
                frame_ms=nearest_ms,
                image=np.asarray(frame.array),
                target_bbox=bbox,
                visible=visible,
            )
        )
    return samples


def _annotation_to_target(
    annotation: YoyoFrameAnnotation,
    point_box_size: float,
) -> tuple[tuple[float, float, float, float], bool]:
    bbox: BoundingBox | None = None
    if annotation.bbox is not None:
        bbox = BoundingBox(
            x=annotation.bbox.x,
            y=annotation.bbox.y,
            width=annotation.bbox.width,
            height=annotation.bbox.height,
        )
    point_x = annotation.point.x if annotation.point is not None else None
    point_y = annotation.point.y if annotation.point is not None else None
    return target_bbox_from_annotation(
        point_x=point_x,
        point_y=point_y,
        bbox=bbox,
        visibility=annotation.visibility,
        point_box_size=point_box_size,
    )


def load_detector_samples_from_dataset(
    dataset_dir: Path,
    config: DetectorTrainingConfig | None = None,
) -> list[DetectorTrainingSample]:
    config = config or DetectorTrainingConfig()
    _manifest, records = load_dataset(dataset_dir)
    selected = [record for record in select_training_records(records) if record.yoyo_track]
    if not selected:
        raise DatasetBridgeError("No selected records contain yoyo_track annotations.")

    samples: list[DetectorTrainingSample] = []
    for record in selected:
        samples.extend(_samples_for_record(dataset_dir, record, config))
    if not samples:
        raise DatasetBridgeError(
            "No training frames matched annotated yoyo_track timestamps. "
            "Check sample_fps / OpenCV availability / video files."
        )
    return samples
