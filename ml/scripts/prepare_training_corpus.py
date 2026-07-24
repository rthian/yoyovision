#!/usr/bin/env python
"""Prepare a training corpus: validate dataset, run perception, train TCN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from yoyovision_ml.dataset.io import load_dataset, select_training_records
from yoyovision_ml.dataset.ontology import default_ontology
from yoyovision_ml.dataset.validators import validate_dataset
from yoyovision_ml.events.checkpoint import EventModelMetadata, save_checkpoint
from yoyovision_ml.events.config import InferenceConfig, TrainingConfig
from yoyovision_ml.events.dataset_bridge import load_training_samples_from_dataset
from yoyovision_ml.events.train import train_model
from yoyovision_ml.perception import pipeline as perception_module  # noqa: F401
from yoyovision_ml.perception.pipeline import PerceptionPipeline


def _validate_dataset_dir(dataset_dir: Path) -> int:
    manifest, records = load_dataset(dataset_dir)
    report = validate_dataset(manifest, records, dataset_dir, default_ontology())
    for issue in report.issues:
        location = " ".join(
            f"[{key}={value}]"
            for key, value in (("video", issue.video_id), ("record", issue.record_id))
            if value is not None
        )
        print(f"{issue.severity.upper():7} {issue.rule.value:32} {location} {issue.message}")
    print(
        f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s). "
        f"Dataset is {'VALID' if report.is_valid else 'INVALID'}."
    )
    return 0 if report.is_valid else 1


def _run_perception(
    dataset_dir: Path,
    perception_dir: Path,
    *,
    pose_adapter: str,
    hand_adapter: str,
    yoyo_adapter: str,
    tracker_adapter: str,
    sample_fps: float,
    tracker_max_gap_ms: int,
    tracker_static_camera: bool,
) -> None:
    _manifest, records = load_dataset(dataset_dir)
    selected = select_training_records(records)
    tracker_kwargs: dict[str, object] = {}
    if tracker_adapter == "kalman":
        tracker_kwargs = {
            "max_gap_ms": tracker_max_gap_ms,
            "static_camera": tracker_static_camera,
        }

    pipeline = PerceptionPipeline(
        pose_adapter_name=pose_adapter,
        hand_adapter_name=hand_adapter,
        yoyo_adapter_name=yoyo_adapter,
        tracker_adapter_name=tracker_adapter,
        tracker_adapter_kwargs=tracker_kwargs,
        sample_fps=sample_fps,
    )
    perception_dir.mkdir(parents=True, exist_ok=True)

    for record in selected:
        video_path = dataset_dir / record.video.relative_path
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video for {record.record_id}: {video_path}")
        print(f"Running perception for {record.video.video_id} ({video_path.name})...")
        parquet_path, _metadata_path = pipeline.run_and_write(
            video_path,
            duration_ms=record.video.duration_ms,
            fps=record.video.source_fps,
            output_dir=perception_dir,
            name=record.video.video_id,
        )
        print(f"  wrote {parquet_path}")


def _train_model(
    dataset_dir: Path,
    perception_dir: Path,
    model_dir: Path,
    train_name: str,
    *,
    feature_subset: str,
    seed: int,
    max_epochs: int,
) -> None:
    samples = load_training_samples_from_dataset(dataset_dir, perception_dir)
    config = TrainingConfig(feature_subset=feature_subset, seed=seed, max_epochs=max_epochs)
    result = train_model(samples, config=config, inference_config=InferenceConfig())
    metadata = EventModelMetadata(
        model_name=result.model_name,
        model_version=result.model_version,
        training_config=result.config,
        feature_names=result.feature_names,
        input_dim=len(result.feature_names),
        normalization=result.normalization.to_dict(),
        calibration_temperatures=result.calibration_temperatures.tolist(),
        player_splits=result.player_splits,
        best_epoch=result.best_epoch,
        val_loss_history=result.val_loss_history,
        val_metrics=result.val_report.to_dict(),
        test_metrics=result.test_report.to_dict() if result.test_report else None,
        train_sample_count=result.train_sample_count,
        val_sample_count=result.val_sample_count,
        test_sample_count=result.test_sample_count,
        torch_version=result.torch_module.__version__,
        training_data_source="dataset",
    )
    weights_path, metadata_path = save_checkpoint(
        result.torch_module, result.model, metadata, model_dir, train_name
    )
    print(f"Wrote checkpoint weights: {weights_path}")
    print(f"Wrote checkpoint metadata: {metadata_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_training_corpus",
        description="Validate a dataset, run perception, and optionally train the TCN.",
    )
    parser.add_argument("dataset_dir")
    parser.add_argument("--perception-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--pose-adapter", default="mock", choices=["mock", "mediapipe"])
    parser.add_argument("--hand-adapter", default="mock", choices=["mock", "mediapipe"])
    parser.add_argument("--yoyo-adapter", default="mock")
    parser.add_argument("--tracker-adapter", default="mock", choices=["mock", "kalman"])
    parser.add_argument("--sample-fps", type=float, default=15.0)
    parser.add_argument("--tracker-max-gap-ms", type=int, default=500)
    parser.add_argument("--tracker-static-camera", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--skip-perception", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--train-name", default="tcn")
    parser.add_argument("--feature-subset", default="fused")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    perception_dir = Path(args.perception_dir or dataset_dir / "perception")

    if not args.skip_validate:
        exit_code = _validate_dataset_dir(dataset_dir)
        if exit_code != 0:
            return exit_code

    if not args.skip_perception:
        _run_perception(
            dataset_dir,
            perception_dir,
            pose_adapter=args.pose_adapter,
            hand_adapter=args.hand_adapter,
            yoyo_adapter=args.yoyo_adapter,
            tracker_adapter=args.tracker_adapter,
            sample_fps=args.sample_fps,
            tracker_max_gap_ms=args.tracker_max_gap_ms,
            tracker_static_camera=args.tracker_static_camera,
        )

    if not args.skip_train:
        if args.model_dir is None:
            print("error: --model-dir is required unless --skip-train is set.", file=sys.stderr)
            return 1
        try:
            _train_model(
                dataset_dir,
                perception_dir,
                Path(args.model_dir),
                args.train_name,
                feature_subset=args.feature_subset,
                seed=args.seed,
                max_epochs=args.max_epochs,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print("Corpus preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
