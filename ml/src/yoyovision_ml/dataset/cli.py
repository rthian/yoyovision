"""Command-line tools for the YoYoVision dataset foundation.

Usage:

    python -m yoyovision_ml.dataset.cli validate <dataset_dir>
    python -m yoyovision_ml.dataset.cli stats <dataset_dir>
    python -m yoyovision_ml.dataset.cli split <dataset_dir> [--seed 42] [--write]
    python -m yoyovision_ml.dataset.cli import-cvat <xml_path> --fps 30 --width 1920 \\
        --height 1080 [--track-label yoyo]

Also installed as the `yoyovision-dataset` console script (see pyproject.toml).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yoyovision_ml.dataset.importers.cvat import CvatImportError, import_cvat_yoyo_track
from yoyovision_ml.dataset.io import load_dataset, load_record, save_manifest, save_record
from yoyovision_ml.dataset.ontology import default_ontology
from yoyovision_ml.dataset.schema import SplitName
from yoyovision_ml.dataset.splits import DEFAULT_RATIOS, generate_player_grouped_splits
from yoyovision_ml.dataset.stats import compute_annotator_agreement, compute_dataset_statistics
from yoyovision_ml.dataset.validators import validate_dataset


def _cmd_validate(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir)
    manifest, records = load_dataset(dataset_dir)
    ontology = default_ontology()
    report = validate_dataset(manifest, records, dataset_dir, ontology)

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


def _cmd_stats(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir)
    manifest, records = load_dataset(dataset_dir)
    statistics = compute_dataset_statistics(records, manifest.splits)

    print(f"videos: {statistics.video_count}")
    print(f"players: {statistics.player_count}")
    print(f"records (annotation passes): {statistics.record_count}")
    print(f"total duration: {statistics.total_duration_ms / 1000:.1f}s")
    print(f"events: {statistics.event_count}")
    print(f"deductions: {statistics.deduction_count}")
    print("events by family:")
    for family, count in statistics.events_by_family.most_common():
        print(f"  {family}: {count}")
    print("events by outcome:")
    for outcome, count in statistics.events_by_outcome.most_common():
        print(f"  {outcome}: {count}")
    print("events by difficulty band:")
    for band, count in statistics.events_by_difficulty_band.most_common():
        print(f"  {band}: {count}")
    if statistics.videos_by_split:
        print("videos by split:")
        for split, count in statistics.videos_by_split.most_common():
            print(f"  {split}: {count}")

    agreements = compute_annotator_agreement(records)
    if agreements:
        print("\nannotator agreement (non-adjudicated passes over the same video):")
        for agreement in agreements:
            print(
                f"  video={agreement.video_id} {agreement.annotator_a} vs "
                f"{agreement.annotator_b}: agreement_ratio={agreement.agreement_ratio} "
                f"outcome_agreement_ratio={agreement.outcome_agreement_ratio}"
            )
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir)
    manifest, records = load_dataset(dataset_dir)
    videos = [r.video for r in records]

    ratios = DEFAULT_RATIOS
    if args.train is not None or args.val is not None or args.test is not None:
        ratios = {
            SplitName.TRAIN: args.train if args.train is not None else 0.7,
            SplitName.VAL: args.val if args.val is not None else 0.15,
            SplitName.TEST: args.test if args.test is not None else 0.15,
        }

    assignment = generate_player_grouped_splits(videos, seed=args.seed, ratios=ratios)

    for video_id, split in sorted(assignment.video_splits.items()):
        print(f"{video_id}\t{split.value}")

    if args.write:
        manifest.splits = dict(assignment.video_splits)
        manifest.split_seed = assignment.seed
        save_manifest(dataset_dir, manifest)
        print(
            f"\nWrote split assignment (seed={assignment.seed}) to manifest.json", file=sys.stderr
        )
    return 0


def _cmd_import_cvat(args: argparse.Namespace) -> int:
    try:
        annotations = import_cvat_yoyo_track(
            Path(args.xml_path),
            fps=args.fps,
            frame_width=args.width,
            frame_height=args.height,
            track_label=args.track_label,
        )
    except CvatImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = [a.model_dump(mode="json") for a in annotations]
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {len(annotations)} frame annotations to {args.output}")
    else:
        print(json.dumps(payload, indent=2))
    return 0




def _cmd_merge_cvat(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir)
    record_path = dataset_dir / "records" / f"{args.record_id}.json"
    if not record_path.exists():
        record_path = Path(args.record_id)
    if not record_path.exists():
        print(f"error: record not found: {args.record_id}", file=sys.stderr)
        return 1

    try:
        annotations = import_cvat_yoyo_track(
            Path(args.xml_path),
            fps=args.fps,
            frame_width=args.width,
            frame_height=args.height,
            track_label=args.track_label,
        )
    except CvatImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    record = load_record(record_path)
    updated = record.model_copy(update={"yoyo_track": annotations})
    out_path = save_record(dataset_dir, updated)
    print(f"Merged {len(annotations)} yo-yo frames into {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yoyovision-dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a dataset directory.")
    validate_parser.add_argument("dataset_dir")
    validate_parser.set_defaults(func=_cmd_validate)

    stats_parser = subparsers.add_parser("stats", help="Print dataset statistics.")
    stats_parser.add_argument("dataset_dir")
    stats_parser.set_defaults(func=_cmd_stats)

    split_parser = subparsers.add_parser("split", help="Generate player-grouped splits.")
    split_parser.add_argument("dataset_dir")
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--train", type=float, default=None)
    split_parser.add_argument("--val", type=float, default=None)
    split_parser.add_argument("--test", type=float, default=None)
    split_parser.add_argument(
        "--write", action="store_true", help="Persist the split assignment into manifest.json."
    )
    split_parser.set_defaults(func=_cmd_split)

    cvat_parser = subparsers.add_parser(
        "import-cvat", help="Import a CVAT video-XML box/point track as a yo-yo track."
    )
    cvat_parser.add_argument("xml_path")
    cvat_parser.add_argument("--fps", type=float, required=True)
    cvat_parser.add_argument("--width", type=int, required=True)
    cvat_parser.add_argument("--height", type=int, required=True)
    cvat_parser.add_argument("--track-label", default="yoyo")
    cvat_parser.add_argument("--output", default=None, help="Write JSON output to this path.")
    cvat_parser.set_defaults(func=_cmd_import_cvat)

    merge_parser = subparsers.add_parser(
        "merge-cvat",
        help="Import a CVAT yo-yo track directly into a DatasetRecord.yoyo_track.",
    )
    merge_parser.add_argument("dataset_dir")
    merge_parser.add_argument("record_id", help="Record id or records/<id>.json basename.")
    merge_parser.add_argument("xml_path")
    merge_parser.add_argument("--fps", type=float, required=True)
    merge_parser.add_argument("--width", type=int, required=True)
    merge_parser.add_argument("--height", type=int, required=True)
    merge_parser.add_argument("--track-label", default="yoyo")
    merge_parser.set_defaults(func=_cmd_merge_cvat)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
