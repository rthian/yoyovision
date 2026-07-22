"""Command-line tools for the Prompt C temporal trick-event model.

Usage:

    python -m yoyovision_ml.events.cli train --output-dir <dir> --name <name> \\
        [--feature-subset fused|skeleton|trajectory] [--seed 42] [--max-epochs 50] \\
        [--window-ms 4000] [--stride-ms 2000] [--hidden-channels 64] [--num-blocks 4] \\
        [--batch-size 16] [--learning-rate 1e-3] [--early-stopping-patience 8] \\
        [--num-players 6] [--clips-per-player 2] [--num-events-per-clip 10]

    python -m yoyovision_ml.events.cli run --detector majority|rules|torch [--weights PATH] \\
        (--features <perception_artifact.parquet> | --synthetic-seed N --synthetic-video-id ID \\
         --synthetic-player-id ID) --output-dir <dir> --name <name>

    python -m yoyovision_ml.events.cli evaluate --predictions <predictions.parquet> \\
        (--record <dataset_record.json> | --synthetic-seed N --synthetic-video-id ID \\
         --synthetic-player-id ID) [--tiou-threshold 0.5]

    python -m yoyovision_ml.events.cli compare-baselines [--seed 42] [--num-players 6] \\
        [--clips-per-player 2] [--num-events-per-clip 10] [--max-epochs 10] \\
        [--hidden-channels 32] [--num-blocks 3] [--split val|test]

    python -m yoyovision_ml.events.cli compare-modalities [--seed 42] [--num-players 6] \\
        [--clips-per-player 2] [--num-events-per-clip 10] [--max-epochs 10] \\
        [--hidden-channels 32] [--num-blocks 3] [--split val|test]

Also installed as the `yoyovision-events` console script (see pyproject.toml).

Every synthetic-data code path here is clearly labelled `training_data_source:
"synthetic"` in its output -- per Prompt C's "Do not claim production
readiness based only on clip-level accuracy", none of these commands'
output should be read as a measurement of real-world model quality until
real annotated footage is wired in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yoyovision_ml.adapters_registry import (
    AdapterNotRegisteredError,
    create_temporal_event_detector,
)
from yoyovision_ml.events.artifact import read_predictions, write_predictions
from yoyovision_ml.events.baselines import MajorityClassEventDetector, ThresholdRuleEventDetector
from yoyovision_ml.events.checkpoint import EventModelMetadata, save_checkpoint
from yoyovision_ml.events.config import InferenceConfig, TrainingConfig
from yoyovision_ml.events.dataset_bridge import DatasetBridgeError, load_training_samples_from_dataset
from yoyovision_ml.events.metrics import evaluate, evaluate_detector
from yoyovision_ml.events.synthetic import generate_synthetic_dataset, generate_synthetic_sample
from yoyovision_ml.events.train import player_grouped_split, train_model
from yoyovision_ml.events.types import TrainingSample
from yoyovision_ml.interfaces import TemporalEventDetector
from yoyovision_ml.perception.errors import (
    MissingOptionalDependencyError,
    ModelWeightsNotConfiguredError,
)


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--feature-subset",
        default="fused",
        choices=["fused", "skeleton", "trajectory", "kinematics_only", "multimodal_fused"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-ms", type=int, default=4000)
    parser.add_argument("--stride-ms", type=int, default=2000)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-players", type=int, default=6)
    parser.add_argument("--clips-per-player", type=int, default=2)
    parser.add_argument("--num-events-per-clip", type=int, default=10)


def _training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        feature_subset=args.feature_subset,
        seed=args.seed,
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


def _cmd_train(args: argparse.Namespace) -> int:
    config = _training_config_from_args(args)
    inference_config = InferenceConfig()
    training_data_source = "synthetic"
    if args.dataset_dir:
        if not args.perception_dir:
            print(
                "error: --perception-dir is required when --dataset-dir is set.",
                file=sys.stderr,
            )
            return 1
        try:
            samples = load_training_samples_from_dataset(
                Path(args.dataset_dir), Path(args.perception_dir)
            )
        except (DatasetBridgeError, FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        training_data_source = "dataset"
    else:
        samples = generate_synthetic_dataset(
            num_players=args.num_players,
            clips_per_player=args.clips_per_player,
            seed=args.seed,
            num_events_per_clip=args.num_events_per_clip,
        )

    try:
        result = train_model(samples, config=config, inference_config=inference_config)
    except (MissingOptionalDependencyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

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
        training_data_source=training_data_source,
    )
    weights_path, metadata_path = save_checkpoint(
        result.torch_module, result.model, metadata, Path(args.output_dir), args.name
    )

    print(f"Wrote checkpoint weights: {weights_path}")
    print(f"Wrote checkpoint metadata: {metadata_path}")
    print(f"best_epoch: {result.best_epoch}  model_version: {result.model_version}")
    print(
        json.dumps(
            {"val_metrics": metadata.val_metrics, "test_metrics": metadata.test_metrics}, indent=2
        )
    )
    return 0


def _synthetic_sample_from_args(args: argparse.Namespace) -> TrainingSample:
    return generate_synthetic_sample(
        seed=args.synthetic_seed,
        video_id=args.synthetic_video_id,
        player_id=args.synthetic_player_id,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    if args.features:
        from yoyovision_ml.perception.artifact import read_artifact

        features, _metadata = read_artifact(Path(args.features))
        video_id = Path(args.features).stem
    elif args.synthetic_seed is not None:
        sample = _synthetic_sample_from_args(args)
        features = sample.features
        video_id = sample.video_id
    else:
        print(
            "error: pass either --features or --synthetic-seed/--synthetic-video-id/"
            "--synthetic-player-id",
            file=sys.stderr,
        )
        return 1

    detector_kwargs: dict[str, object] = {}
    if args.detector == "torch" and args.weights:
        detector_kwargs["weights_path"] = args.weights

    try:
        detector: TemporalEventDetector = create_temporal_event_detector(  # type: ignore[assignment]
            args.detector, **detector_kwargs
        )
        predictions, _deductions = detector.predict(features)
    except (
        AdapterNotRegisteredError,
        MissingOptionalDependencyError,
        ModelWeightsNotConfiguredError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parquet_path, metadata_path = write_predictions(
        predictions, video_id, Path(args.output_dir), args.name
    )
    print(f"Wrote {len(predictions)} predictions to: {parquet_path}")
    print(f"Wrote metadata: {metadata_path}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    predictions, _metadata = read_predictions(Path(args.predictions))

    if args.record:
        from yoyovision_ml.dataset.io import load_record

        record = load_record(Path(args.record))
        ground_truth = list(record.trick_events)
    elif args.synthetic_seed is not None:
        ground_truth = list(_synthetic_sample_from_args(args).trick_events)
    else:
        print(
            "error: pass either --record or --synthetic-seed/--synthetic-video-id/"
            "--synthetic-player-id",
            file=sys.stderr,
        )
        return 1

    report = evaluate(predictions, ground_truth, tiou_threshold=args.tiou_threshold)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def _cmd_compare_baselines(args: argparse.Namespace) -> int:
    """Prompt C: "Provide a comparison against simple baselines: majority
    class, hand-crafted threshold rules, skeleton-only model, yo-yo-
    trajectory-only model, fused model." All five run against the *same*
    player-grouped split of one synthetic dataset (`player_grouped_split`
    depends only on `(samples, seed, train_ratio, val_ratio)`, never on
    `feature_subset`, so the three trained ablations and the two baselines
    all see identical splits), so the comparison is apples-to-apples."""
    base_config = TrainingConfig(
        seed=args.seed,
        max_epochs=args.max_epochs,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
    )
    samples = generate_synthetic_dataset(
        num_players=args.num_players,
        clips_per_player=args.clips_per_player,
        seed=args.seed,
        num_events_per_clip=args.num_events_per_clip,
    )
    train_samples, val_samples, test_samples = player_grouped_split(
        samples, base_config.seed, base_config.train_ratio, base_config.val_ratio
    )
    eval_samples = test_samples if args.split == "test" else val_samples
    if not eval_samples:
        print(
            f"error: the {args.split} split is empty for this dataset size/seed.", file=sys.stderr
        )
        return 1

    reports: dict[str, object] = {}

    majority_detector = MajorityClassEventDetector.fit(train_samples)
    reports["majority"] = evaluate_detector(majority_detector, eval_samples).to_dict()
    reports["rules"] = evaluate_detector(ThresholdRuleEventDetector(), eval_samples).to_dict()

    for feature_subset in ("skeleton", "trajectory", "fused"):
        config = base_config.model_copy(update={"feature_subset": feature_subset})
        try:
            result = train_model(samples, config=config)
        except MissingOptionalDependencyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.split == "test" and result.test_report is not None:
            report = result.test_report
        else:
            report = result.val_report
        reports[feature_subset] = report.to_dict()

    print(json.dumps(reports, indent=2))
    print(
        "\nNote (Prompt C): this is a small-sample, synthetic-data comparison meant to "
        "exercise the pipeline end to end -- it is not a measurement of real-world "
        "model quality and must not be read as production readiness evidence.",
        file=sys.stderr,
    )
    return 0


def _cmd_compare_modalities(args: argparse.Namespace) -> int:
    """Prompt E: "an ablation study comparing kinematics-only vs.
    RGB/string/audio-fused models." Trains `"kinematics_only"` and
    `"multimodal_fused"` (see `events.labels.FEATURE_SUBSETS`) against the
    *same* player-grouped split of one synthetic dataset -- same
    apples-to-apples rationale as `_cmd_compare_baselines` above. There is
    no real annotated footage with RGB/string-mask/audio ground truth in
    this repository (same situation Prompt C's `compare-baselines` already
    documents for kinematic-only ablations), so any metric delta here is a
    synthetic-data artifact of `events.synthetic`'s generator treating every
    feature family symmetrically -- evidence the tooling works end to end,
    not evidence about real-world multimodal fusion value.
    """
    base_config = TrainingConfig(
        seed=args.seed,
        max_epochs=args.max_epochs,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
    )
    samples = generate_synthetic_dataset(
        num_players=args.num_players,
        clips_per_player=args.clips_per_player,
        seed=args.seed,
        num_events_per_clip=args.num_events_per_clip,
    )
    train_samples, val_samples, test_samples = player_grouped_split(
        samples, base_config.seed, base_config.train_ratio, base_config.val_ratio
    )
    eval_samples = test_samples if args.split == "test" else val_samples
    if not eval_samples:
        print(
            f"error: the {args.split} split is empty for this dataset size/seed.", file=sys.stderr
        )
        return 1

    reports: dict[str, object] = {}
    for feature_subset in ("kinematics_only", "multimodal_fused"):
        config = base_config.model_copy(update={"feature_subset": feature_subset})
        try:
            result = train_model(samples, config=config)
        except MissingOptionalDependencyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.split == "test" and result.test_report is not None:
            report = result.test_report
        else:
            report = result.val_report
        reports[feature_subset] = report.to_dict()

    print(json.dumps(reports, indent=2))
    print(
        "\nNote (Prompt E): this is a small-sample, synthetic-data comparison meant to "
        "exercise the multimodal fusion pipeline/tooling end to end -- there is no real "
        "annotated footage with RGB/string-mask/audio ground truth yet, so this must not "
        "be read as evidence that RGB/string/audio fusion improves real-world model "
        "quality.",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yoyovision-events")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train the temporal trick-event TCN on synthetic or dataset clips."
    )
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--name", required=True)
    train_parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Prompt A dataset directory with records/ and manifest.json.",
    )
    train_parser.add_argument(
        "--perception-dir",
        default=None,
        help="Directory of perception Parquet artefacts keyed by video_id.",
    )
    _add_training_arguments(train_parser)
    train_parser.set_defaults(func=_cmd_train)

    run_parser = subparsers.add_parser(
        "run", help="Run a TemporalEventDetector adapter on one clip's features."
    )
    run_parser.add_argument("--detector", required=True, choices=["majority", "rules", "torch"])
    run_parser.add_argument(
        "--weights", default=None, help="Checkpoint (.pt) path for --detector torch."
    )
    run_parser.add_argument(
        "--features", default=None, help="Perception feature artifact (.parquet)."
    )
    run_parser.add_argument("--synthetic-seed", type=int, default=None)
    run_parser.add_argument("--synthetic-video-id", default="synthetic-cli-video")
    run_parser.add_argument("--synthetic-player-id", default="synthetic-cli-player")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--name", required=True)
    run_parser.set_defaults(func=_cmd_run)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Score a predictions artifact against ground truth."
    )
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.add_argument(
        "--record", default=None, help="Dataset record (.json) with trick_events."
    )
    evaluate_parser.add_argument("--synthetic-seed", type=int, default=None)
    evaluate_parser.add_argument("--synthetic-video-id", default="synthetic-cli-video")
    evaluate_parser.add_argument("--synthetic-player-id", default="synthetic-cli-player")
    evaluate_parser.add_argument("--tiou-threshold", type=float, default=0.5)
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    compare_parser = subparsers.add_parser(
        "compare-baselines",
        help="Compare majority/rules/skeleton-only/trajectory-only/fused on one split.",
    )
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.add_argument("--num-players", type=int, default=6)
    compare_parser.add_argument("--clips-per-player", type=int, default=2)
    compare_parser.add_argument("--num-events-per-clip", type=int, default=10)
    compare_parser.add_argument("--max-epochs", type=int, default=10)
    compare_parser.add_argument("--hidden-channels", type=int, default=32)
    compare_parser.add_argument("--num-blocks", type=int, default=3)
    compare_parser.add_argument("--split", default="test", choices=["val", "test"])
    compare_parser.set_defaults(func=_cmd_compare_baselines)

    compare_modalities_parser = subparsers.add_parser(
        "compare-modalities",
        help="Prompt E ablation: compare kinematics-only vs. multimodal-fused (RGB/"
        "string-seg/audio) on one synthetic split.",
    )
    compare_modalities_parser.add_argument("--seed", type=int, default=42)
    compare_modalities_parser.add_argument("--num-players", type=int, default=6)
    compare_modalities_parser.add_argument("--clips-per-player", type=int, default=2)
    compare_modalities_parser.add_argument("--num-events-per-clip", type=int, default=10)
    compare_modalities_parser.add_argument("--max-epochs", type=int, default=10)
    compare_modalities_parser.add_argument("--hidden-channels", type=int, default=32)
    compare_modalities_parser.add_argument("--num-blocks", type=int, default=3)
    compare_modalities_parser.add_argument("--split", default="test", choices=["val", "test"])
    compare_modalities_parser.set_defaults(func=_cmd_compare_modalities)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
