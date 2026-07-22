"""Prompt D scoring/calibration command-line tools. Tests `scoring.cli`
(the `yoyovision-scoring` console script): the `score` and `calibrate`
subcommands, end-to-end over synthetic dataset records and prediction
artifacts (no production data)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import make_event_prediction

from yoyovision_ml.dataset.schema import (
    AnnotationProvenance,
    DatasetRecord,
    DatasetVideo,
    DeductionAnnotation,
    FreestyleEvaluationAnnotation,
    JudgeClickAnnotation,
    TrickEventAnnotation,
)
from yoyovision_ml.domain import DeductionType, DifficultyBand, EventFamily, Outcome, Source
from yoyovision_ml.events.artifact import write_predictions
from yoyovision_ml.scoring import cli
from yoyovision_ml.scoring.profiles import ScoringProfile


def _provenance(*, is_adjudicated: bool = False) -> AnnotationProvenance:
    return AnnotationProvenance(
        annotator_id="annotator-1",
        source=Source.HUMAN,
        annotated_at=datetime(2026, 1, 1, tzinfo=UTC),
        is_adjudicated=is_adjudicated,
        adjudicated_by="reviewer-1" if is_adjudicated else None,
    )


def _dataset_video() -> DatasetVideo:
    return DatasetVideo(
        video_id="video-1",
        player_id="player-1",
        relative_path="videos/video-1.mp4",
        checksum_sha256="0" * 64,
        duration_ms=10_000,
        width=1920,
        height=1080,
        source_fps=30.0,
    )


def _make_record(
    *,
    record_id: str = "record-1",
    is_adjudicated: bool = False,
    trick_events: list[TrickEventAnnotation] | None = None,
    deductions: list[DeductionAnnotation] | None = None,
    judge_clicks: list[JudgeClickAnnotation] | None = None,
    freestyle_evaluations: list[FreestyleEvaluationAnnotation] | None = None,
) -> DatasetRecord:
    return DatasetRecord(
        record_id=record_id,
        video=_dataset_video(),
        annotator_id="annotator-1",
        is_adjudicated=is_adjudicated,
        ontology_version="test-ontology-1",
        trick_events=trick_events or [],
        deductions=deductions or [],
        judge_clicks=judge_clicks or [],
        freestyle_evaluations=freestyle_evaluations or [],
    )


def _write_record(path: Path, record: DatasetRecord) -> Path:
    path.write_text(record.model_dump_json(), encoding="utf-8")
    return path


def test_build_parser_score_defaults() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["score", "--record", "record.json"])
    assert args.profile == ScoringProfile.JUDGE_ASSIST.value
    assert args.ruleset_version is None
    assert args.bootstrap_iterations == 500
    assert args.bootstrap_seed == 0


def test_build_parser_calibrate_defaults() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["calibrate"])
    assert args.pair == []
    assert args.click_tolerance_ms == 1000
    assert args.plot_output is None


def test_build_parser_score_rejects_unknown_profile() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["score", "--record", "record.json", "--profile", "not-a-profile"])


def test_cmd_score_end_to_end_prints_valid_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _make_record(
        trick_events=[
            TrickEventAnnotation(
                event_id="evt-1",
                label="mount_1",
                family=EventFamily.MOUNT,
                start_ms=0,
                end_ms=500,
                outcome=Outcome.SUCCESS,
                difficulty_band=DifficultyBand.BASIC,
                provenance=_provenance(),
            )
        ],
        deductions=[
            DeductionAnnotation(
                deduction_id="ded-1",
                type=DeductionType.YOYO_STOP,
                timestamp_ms=1000,
                provenance=_provenance(),
            )
        ],
    )
    record_path = _write_record(tmp_path / "record.json", record)

    exit_code = cli.main(["score", "--record", str(record_path)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "judge_assist"
    assert payload["technical_event_count"] == 1
    assert payload["major_deductions"] > 0.0
    assert "breakdown" in payload


def test_cmd_score_respects_profile_argument(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _make_record()
    record_path = _write_record(tmp_path / "record.json", record)

    exit_code = cli.main(["score", "--record", str(record_path), "--profile", "research"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "research"


def test_cmd_score_unknown_ruleset_version_returns_error_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _make_record()
    record_path = _write_record(tmp_path / "record.json", record)

    exit_code = cli.main(
        ["score", "--record", str(record_path), "--ruleset-version", "does-not-exist"]
    )

    assert exit_code == 1
    assert "error" in capsys.readouterr().err.lower()


def test_cmd_score_pending_annotation_still_counts_under_judge_assist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-adjudicated annotation (`is_adjudicated=False`) becomes a
    PENDING `AnalysisEvent`, which `judge_assist` (the CLI's default
    profile) still counts -- mirrors `_event_review_status`'s behavior."""
    record = _make_record(
        trick_events=[
            TrickEventAnnotation(
                event_id="evt-1",
                label="mount_1",
                family=EventFamily.MOUNT,
                start_ms=0,
                end_ms=500,
                outcome=Outcome.SUCCESS,
                provenance=_provenance(is_adjudicated=False),
            )
        ]
    )
    record_path = _write_record(tmp_path / "record.json", record)

    cli.main(["score", "--record", str(record_path)])

    payload = json.loads(capsys.readouterr().out)
    assert payload["technical_event_count"] == 1


def test_cmd_calibrate_requires_at_least_one_pair(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["calibrate"])
    assert exit_code == 1
    assert "--pair" in capsys.readouterr().err


def test_cmd_calibrate_unknown_ruleset_version_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _make_record()
    record_path = _write_record(tmp_path / "record.json", record)
    pred_path, _ = write_predictions([], "video-1", tmp_path, "preds")

    exit_code = cli.main(
        [
            "calibrate",
            "--pair",
            str(record_path),
            str(pred_path),
            "--ruleset-version",
            "does-not-exist",
        ]
    )
    assert exit_code == 1


def test_cmd_calibrate_end_to_end_reports_agreement_statistics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ground_truth_record = _make_record(
        trick_events=[
            TrickEventAnnotation(
                event_id="evt-1",
                label="mount_1",
                family=EventFamily.MOUNT,
                start_ms=0,
                end_ms=500,
                outcome=Outcome.SUCCESS,
                difficulty_band=DifficultyBand.BASIC,
                provenance=_provenance(is_adjudicated=True),
            )
        ],
        judge_clicks=[
            JudgeClickAnnotation(click_id="c1", judge_id="judge-a", timestamp_ms=0),
        ],
        freestyle_evaluations=[
            FreestyleEvaluationAnnotation(
                judge_id="judge-a",
                execution=7.0,
                control=7.0,
                trick_diversity=7.0,
                space_use_emphasis=7.0,
                music_choreography=7.0,
                music_construction=7.0,
                body_control=7.0,
                showmanship=7.0,
                provenance=_provenance(is_adjudicated=True),
            )
        ],
    )
    record_path = _write_record(tmp_path / "record.json", ground_truth_record)

    predictions = [make_event_prediction(label="mount_1", start_ms=0)]
    pred_path, _ = write_predictions(predictions, "video-1", tmp_path, "preds")

    exit_code = cli.main(
        [
            "calibrate",
            "--pair",
            str(record_path),
            str(pred_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_records"] == 1
    assert len(payload["model_scores"]) == 1
    assert len(payload["judge_scores"]) == 1
    assert "score_agreement" in payload
    assert payload["event_count_agreement"][0]["record_id"] == "record-1"


def test_cmd_calibrate_with_plot_output_writes_a_png(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("matplotlib")
    record = _make_record(
        trick_events=[
            TrickEventAnnotation(
                event_id="evt-1",
                label="mount_1",
                family=EventFamily.MOUNT,
                start_ms=0,
                end_ms=500,
                outcome=Outcome.SUCCESS,
                provenance=_provenance(is_adjudicated=True),
            )
        ],
    )
    record_path = _write_record(tmp_path / "record.json", record)
    pred_path, _ = write_predictions(
        [make_event_prediction(label="mount_1", start_ms=0)], "video-1", tmp_path, "preds"
    )
    plot_path = tmp_path / "cal.png"

    exit_code = cli.main(
        [
            "calibrate",
            "--pair",
            str(record_path),
            str(pred_path),
            "--plot-output",
            str(plot_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plot_path"] == str(plot_path)
    assert plot_path.exists()
