"""Command-line tools for Prompt D scoring and judge calibration.

Usage:

    python -m yoyovision_ml.scoring.cli score --record <dataset_record.json> \\
        [--profile practice|judge_assist|research] [--ruleset-version <version>] \\
        [--bootstrap-iterations 500] [--bootstrap-seed 0]

    python -m yoyovision_ml.scoring.cli calibrate \\
        --pair <dataset_record.json> <predictions.parquet> [--pair ... ...] \\
        [--ruleset-version <version>] [--click-tolerance-ms 1000] [--plot-output <file.png>]

Also installed as the `yoyovision-scoring` console script (see pyproject.toml).

`score` runs the full `scoring.pipeline.run_scoring_pipeline` over one
`dataset.schema.DatasetRecord`'s own annotations (trick events, deductions,
judge clicks, judge Freestyle Evaluations) -- useful for smoke-testing a
ruleset/profile against real or synthetic annotation data without touching
`api`'s database.

`calibrate` implements Prompt D's CALIBRATION section: "compare model
output with expert judges using: mean absolute error, Spearman rank
correlation, Pearson correlation, intraclass correlation where appropriate,
event-count precision and recall, Bland-Altman-style error summaries, score
calibration plots." For each `--pair`, the "expert" reference score is
computed from the record's own (adjudicated) ground-truth trick events/
deductions plus its judge Freestyle Evaluations (profile `research`); the
"model" score is computed from a paired `events.cli run` predictions
artifact with no human input at all (profile `judge_assist`, automatic FE
estimators only). Comparing these two is only as good as the ground-truth
annotations supplied -- this command draws no conclusions about production
readiness from a small sample (same caveat as `events.cli compare-baselines`).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from yoyovision_ml.dataset.io import load_record
from yoyovision_ml.dataset.schema import (
    DeductionAnnotation,
    FreestyleEvaluationAnnotation,
    JudgeClickAnnotation,
    TrickEventAnnotation,
)
from yoyovision_ml.domain import (
    AnalysisEvent,
    AnalysisEventPrediction,
    MajorDeduction,
    ReviewStatus,
    Source,
)
from yoyovision_ml.events.artifact import read_predictions
from yoyovision_ml.ruleset import Ruleset, default_ruleset, get_ruleset_by_version
from yoyovision_ml.scoring.calibration import (
    event_count_agreement,
    paired_agreement,
    paired_agreement_to_dict,
    render_calibration_plot,
)
from yoyovision_ml.scoring.pipeline import run_scoring_pipeline
from yoyovision_ml.scoring.profiles import ScoringProfile
from yoyovision_ml.scoring.types import JudgeClick, JudgeFreestyleScore, ScoringPipelineResult


def _resolve_ruleset(version: str | None) -> Ruleset:
    if version is None:
        return default_ruleset()
    ruleset = get_ruleset_by_version(version)
    if ruleset is None:
        raise ValueError(f"No packaged ruleset with version '{version}'.")
    return ruleset


def _event_review_status(ann: TrickEventAnnotation, *, force_confirmed: bool) -> ReviewStatus:
    if force_confirmed or ann.provenance.is_adjudicated:
        return ReviewStatus.CONFIRMED
    return ReviewStatus.PENDING


def _annotation_to_analysis_event(
    ann: TrickEventAnnotation, *, force_confirmed: bool = False
) -> AnalysisEvent:
    """Converts a dataset `TrickEventAnnotation` into the persisted-shape
    `AnalysisEvent` `scoring.pipeline.run_scoring_pipeline` consumes.
    `force_confirmed` is used by `calibrate`, where the record's events are
    being treated as ground truth regardless of their own adjudication flag
    -- see module docstring."""
    return AnalysisEvent(
        id=ann.event_id,
        analysis_id="dataset-record",
        label=ann.label,
        family=ann.family,
        start_ms=ann.start_ms,
        end_ms=ann.end_ms,
        confidence=ann.confidence,
        outcome=ann.outcome,
        difficulty_band=ann.difficulty_band,
        source=ann.provenance.source,
        review_status=_event_review_status(ann, force_confirmed=force_confirmed),
        model_name=ann.provenance.tool if ann.provenance.source == Source.MODEL else None,
        model_version=ann.provenance.tool_version,
    )


def _annotation_to_major_deduction(
    ann: DeductionAnnotation, *, force_confirmed: bool = False
) -> MajorDeduction:
    review_status = (
        ReviewStatus.CONFIRMED
        if force_confirmed or ann.provenance.is_adjudicated
        else ReviewStatus.PENDING
    )
    return MajorDeduction(
        id=ann.deduction_id,
        analysis_id="dataset-record",
        type=ann.type,
        timestamp_ms=ann.timestamp_ms,
        quantity=ann.quantity,
        # Prospective points, not the scored amount -- scoring_engine
        # recomputes the actually-applied total from `deduction_is_scorable`
        # survivors, ignoring this field entirely. See `MajorDeduction.points`.
        points=0.0,
        confidence=ann.confidence,
        source=ann.provenance.source,
        review_status=review_status,
    )


def _annotation_to_judge_click(ann: JudgeClickAnnotation) -> JudgeClick:
    return JudgeClick(
        click_id=ann.click_id,
        judge_id=ann.judge_id,
        timestamp_ms=ann.timestamp_ms,
        associated_label=ann.associated_label,
        notes=ann.notes,
    )


def _annotation_to_judge_score(ann: FreestyleEvaluationAnnotation) -> JudgeFreestyleScore:
    return JudgeFreestyleScore(
        judge_id=ann.judge_id,
        execution=ann.execution,
        control=ann.control,
        trick_diversity=ann.trick_diversity,
        space_use_emphasis=ann.space_use_emphasis,
        music_choreography=ann.music_choreography,
        music_construction=ann.music_construction,
        body_control=ann.body_control,
        showmanship=ann.showmanship,
        notes=ann.notes,
    )


def _prediction_to_analysis_event(prediction: AnalysisEventPrediction, index: int) -> AnalysisEvent:
    """A freshly-detected, not-yet-reviewed model prediction, in the
    persisted `AnalysisEvent` shape `run_scoring_pipeline` requires."""
    return AnalysisEvent(
        id=f"pred-{index}",
        analysis_id="model-predictions",
        label=prediction.label,
        family=prediction.family,
        start_ms=prediction.start_ms,
        end_ms=prediction.end_ms,
        confidence=prediction.confidence,
        outcome=prediction.outcome,
        difficulty_band=prediction.difficulty_band,
        source=Source.MODEL,
        review_status=ReviewStatus.PENDING,
        model_name=prediction.model_name,
        model_version=prediction.model_version,
    )


def _pipeline_result_to_dict(result: ScoringPipelineResult) -> dict[str, object]:
    return asdict(result)


def _cmd_score(args: argparse.Namespace) -> int:
    try:
        ruleset = _resolve_ruleset(args.ruleset_version)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    record = load_record(Path(args.record))
    events = [_annotation_to_analysis_event(e) for e in record.trick_events]
    deductions = [_annotation_to_major_deduction(d) for d in record.deductions]
    judge_scores = [_annotation_to_judge_score(j) for j in record.freestyle_evaluations]

    result = run_scoring_pipeline(
        events=events,
        deductions=deductions,
        ruleset=ruleset,
        profile=ScoringProfile(args.profile),
        judge_scores=judge_scores,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(_pipeline_result_to_dict(result), indent=2))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    try:
        ruleset = _resolve_ruleset(args.ruleset_version)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.pair:
        print(
            "error: at least one --pair <record.json> <predictions.parquet> is required.",
            file=sys.stderr,
        )
        return 1

    model_scores: list[float] = []
    judge_scores: list[float] = []
    event_count_reports: list[dict[str, object]] = []

    for record_path, predictions_path in args.pair:
        record = load_record(Path(record_path))
        predictions, _metadata = read_predictions(Path(predictions_path))

        ground_truth_events = [
            _annotation_to_analysis_event(e, force_confirmed=True) for e in record.trick_events
        ]
        ground_truth_deductions = [
            _annotation_to_major_deduction(d, force_confirmed=True) for d in record.deductions
        ]
        judge_fe_scores = [_annotation_to_judge_score(j) for j in record.freestyle_evaluations]
        expert_result = run_scoring_pipeline(
            events=ground_truth_events,
            deductions=ground_truth_deductions,
            ruleset=ruleset,
            profile=ScoringProfile.RESEARCH,
            judge_scores=judge_fe_scores,
            bootstrap_iterations=0,
        )

        model_events = [_prediction_to_analysis_event(p, i) for i, p in enumerate(predictions)]
        model_result = run_scoring_pipeline(
            events=model_events,
            deductions=[],
            ruleset=ruleset,
            profile=ScoringProfile.JUDGE_ASSIST,
            bootstrap_iterations=0,
        )

        model_scores.append(model_result.breakdown.final_score)
        judge_scores.append(expert_result.breakdown.final_score)

        clicks = [_annotation_to_judge_click(c) for c in record.judge_clicks]
        agreement = event_count_agreement(predictions, clicks, tolerance_ms=args.click_tolerance_ms)
        event_count_reports.append({"record_id": record.record_id, **asdict(agreement)})

    if not model_scores:
        print("error: no valid record/predictions pairs to calibrate.", file=sys.stderr)
        return 1

    score_agreement = paired_agreement(model_scores, judge_scores)
    report: dict[str, object] = {
        "n_records": len(model_scores),
        "model_scores": model_scores,
        "judge_scores": judge_scores,
        "score_agreement": paired_agreement_to_dict(score_agreement),
        "event_count_agreement": event_count_reports,
    }

    if args.plot_output:
        plot_path = render_calibration_plot(model_scores, judge_scores, Path(args.plot_output))
        report["plot_path"] = str(plot_path)

    print(json.dumps(report, indent=2))
    print(
        "\nNote (Prompt D): a small-sample calibration run is a sanity check, not "
        "production-readiness evidence -- treat it the same way as any other "
        "small-n comparison in this repository.",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yoyovision-scoring")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser(
        "score", help="Score one dataset record with a chosen scoring profile."
    )
    score_parser.add_argument("--record", required=True, help="Dataset record (.json) path.")
    score_parser.add_argument(
        "--profile",
        default=ScoringProfile.JUDGE_ASSIST.value,
        choices=[p.value for p in ScoringProfile],
    )
    score_parser.add_argument("--ruleset-version", default=None)
    score_parser.add_argument("--bootstrap-iterations", type=int, default=500)
    score_parser.add_argument("--bootstrap-seed", type=int, default=0)
    score_parser.set_defaults(func=_cmd_score)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Compare model predictions against expert-judged dataset records.",
    )
    calibrate_parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("RECORD_JSON", "PREDICTIONS_PARQUET"),
        action="append",
        default=[],
        help="A dataset record and its paired model-predictions artifact. Repeatable.",
    )
    calibrate_parser.add_argument("--ruleset-version", default=None)
    calibrate_parser.add_argument("--click-tolerance-ms", type=int, default=1000)
    calibrate_parser.add_argument(
        "--plot-output", default=None, help="Optional .png path for a calibration scatter plot."
    )
    calibrate_parser.set_defaults(func=_cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
