"""Command-line tools for the YoYoVision perception pipeline (Prompt B).

Usage:

    python -m yoyovision_ml.perception.cli run <video_path> --duration-ms 20000 --fps 30 \\
        [--pose-adapter mock|mediapipe] [--hand-adapter mock|mediapipe] \\
        [--yoyo-adapter mock|pytorch|onnx] [--yoyo-weights PATH] \\
        [--tracker-adapter mock|kalman] [--tracker-max-gap-ms 500] [--static-camera] \\
        [--sample-fps 15] --output-dir <dir> --name <artifact_name>

    python -m yoyovision_ml.perception.cli evaluate <video_path> <record_json_path> \\
        --duration-ms 20000 --fps 30 [adapter options as above]

    python -m yoyovision_ml.perception.cli overlay <video_path> --duration-ms 20000 --fps 30 \\
        --output <overlay.mp4> [adapter options as above]

Also installed as the `yoyovision-perception` console script (see pyproject.toml).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

from yoyovision_ml.perception.artifact import metadata_to_dict
from yoyovision_ml.perception.errors import (
    MissingOptionalDependencyError,
    ModelWeightsNotConfiguredError,
)
from yoyovision_ml.perception.evaluation import (
    _YoyoFrameAnnotationLike,
    centre_point_pixel_error,
    detector_precision_recall,
    ground_truth_from_dataset_track,
    interpolation_rate,
    longest_missing_interval,
    normalized_centre_error,
    track_coverage,
)
from yoyovision_ml.perception.overlay import render_overlay_video
from yoyovision_ml.perception.pipeline import PerceptionPipeline


def _build_pipeline(args: argparse.Namespace) -> PerceptionPipeline:
    yoyo_kwargs: dict[str, object] = {}
    if args.yoyo_adapter == "pytorch" and args.yoyo_weights:
        yoyo_kwargs["weights_path"] = args.yoyo_weights
    elif args.yoyo_adapter == "onnx" and args.yoyo_weights:
        yoyo_kwargs["model_path"] = args.yoyo_weights

    tracker_kwargs: dict[str, object] = {}
    if args.tracker_adapter == "kalman":
        tracker_kwargs["max_gap_ms"] = args.tracker_max_gap_ms
        tracker_kwargs["static_camera"] = args.static_camera

    return PerceptionPipeline(
        pose_adapter_name=args.pose_adapter,
        hand_adapter_name=args.hand_adapter,
        yoyo_adapter_name=args.yoyo_adapter,
        tracker_adapter_name=args.tracker_adapter,
        yoyo_adapter_kwargs=yoyo_kwargs,
        tracker_adapter_kwargs=tracker_kwargs,
        sample_fps=args.sample_fps,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        pipeline = _build_pipeline(args)
        _, parquet_path, metadata_path = pipeline.run_and_write(
            Path(args.video_path),
            duration_ms=args.duration_ms,
            fps=args.fps,
            output_dir=Path(args.output_dir),
            name=args.name,
        )
    except (MissingOptionalDependencyError, ModelWeightsNotConfiguredError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote feature table: {parquet_path}")
    print(f"Wrote metadata:      {metadata_path}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from yoyovision_ml.dataset.io import load_record

    try:
        pipeline = _build_pipeline(args)
        result = pipeline.run(Path(args.video_path), duration_ms=args.duration_ms, fps=args.fps)
    except (MissingOptionalDependencyError, ModelWeightsNotConfiguredError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    record = load_record(Path(args.record_json_path))
    ground_truth = ground_truth_from_dataset_track(
        cast(list[_YoyoFrameAnnotationLike], list(record.yoyo_track))
    )
    tracks = list(result.yoyo_tracks)
    expected_frame_ms = [g.frame_ms for g in ground_truth]

    precision_recall = detector_precision_recall(list(result.yoyo_detections), ground_truth)
    report = {
        "video": Path(args.video_path).name,
        "record": Path(args.record_json_path).name,
        "detector_precision_recall": asdict(precision_recall),
        "track_coverage": track_coverage(tracks, expected_frame_ms),
        "longest_missing_interval_ms": longest_missing_interval(tracks, expected_frame_ms),
        "interpolation_rate": interpolation_rate(tracks),
        "normalized_centre_error": asdict(normalized_centre_error(tracks, ground_truth)),
        "centre_point_pixel_error": asdict(
            centre_point_pixel_error(
                tracks, ground_truth, width=record.video.width, height=record.video.height
            )
        ),
    }

    print(json.dumps(report, indent=2))
    return 0


def _cmd_overlay(args: argparse.Namespace) -> int:
    try:
        pipeline = _build_pipeline(args)
        result = pipeline.run(Path(args.video_path), duration_ms=args.duration_ms, fps=args.fps)
        output_path = render_overlay_video(
            Path(args.video_path),
            Path(args.output),
            tracks=list(result.yoyo_tracks),
            pose_sequence=result.pose_sequence,
            hand_sequence=result.hand_sequence,
        )
    except (MissingOptionalDependencyError, ModelWeightsNotConfiguredError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote overlay video: {output_path}")
    print(json.dumps(metadata_to_dict(result.metadata), indent=2))
    return 0


def _add_adapter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--duration-ms", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--sample-fps", type=float, default=15.0)
    parser.add_argument("--pose-adapter", default="mock", choices=["mock", "mediapipe"])
    parser.add_argument("--hand-adapter", default="mock", choices=["mock", "mediapipe"])
    parser.add_argument("--yoyo-adapter", default="mock", choices=["mock", "pytorch", "onnx"])
    parser.add_argument(
        "--yoyo-weights", default=None, help="Checkpoint (.pt) or model (.onnx) path."
    )
    parser.add_argument("--tracker-adapter", default="mock", choices=["mock", "kalman"])
    parser.add_argument("--tracker-max-gap-ms", type=int, default=500)
    parser.add_argument("--static-camera", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yoyovision-perception")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the perception pipeline on one video.")
    run_parser.add_argument("video_path")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--name", required=True)
    _add_adapter_arguments(run_parser)
    run_parser.set_defaults(func=_cmd_run)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run the pipeline and compare tracks/detections against ground truth."
    )
    evaluate_parser.add_argument("video_path")
    evaluate_parser.add_argument("record_json_path")
    _add_adapter_arguments(evaluate_parser)
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    overlay_parser = subparsers.add_parser(
        "overlay", help="Render a debug overlay video (requires OpenCV)."
    )
    overlay_parser.add_argument("video_path")
    overlay_parser.add_argument("--output", required=True)
    _add_adapter_arguments(overlay_parser)
    overlay_parser.set_defaults(func=_cmd_overlay)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
