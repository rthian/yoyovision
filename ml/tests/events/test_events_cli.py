from __future__ import annotations

import json
from pathlib import Path

import pytest

from yoyovision_ml.events.cli import main

torch = pytest.importorskip("torch")

_TINY_TRAIN_ARGS = [
    "--hidden-channels", "4",
    "--num-blocks", "1",
    "--max-epochs", "1",
    "--early-stopping-patience", "1",
    "--window-ms", "1500",
    "--stride-ms", "1500",
    "--num-players", "6",
    "--clips-per-player", "1",
    "--num-events-per-clip", "3",
    "--feature-subset", "trajectory",
]


def test_train_command_writes_checkpoint_and_prints_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "checkpoints"
    exit_code = main(
        ["train", "--output-dir", str(output_dir), "--name", "model", *_TINY_TRAIN_ARGS]
    )

    assert exit_code == 0
    assert (output_dir / "model.pt").exists()
    assert (output_dir / "model.json").exists()

    metadata = json.loads((output_dir / "model.json").read_text())
    assert metadata["training_data_source"] == "synthetic"

    captured = capsys.readouterr()
    assert "Wrote checkpoint weights" in captured.out
    assert "best_epoch" in captured.out


def test_run_command_with_majority_detector_writes_predictions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "predictions"
    exit_code = main(
        [
            "run",
            "--detector", "majority",
            "--synthetic-seed", "1",
            "--synthetic-video-id", "cli-video",
            "--synthetic-player-id", "cli-player",
            "--output-dir", str(output_dir),
            "--name", "preds",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "preds.parquet").exists()
    assert (output_dir / "preds.json").exists()
    assert "Wrote" in capsys.readouterr().out


def test_run_command_requires_either_features_or_synthetic_args(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "run",
            "--detector", "majority",
            "--output-dir", str(tmp_path),
            "--name", "preds",
        ]
    )
    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


def test_run_command_reports_missing_torch_weights_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "run",
            "--detector", "torch",
            "--weights", str(tmp_path / "missing.pt"),
            "--synthetic-seed", "1",
            "--synthetic-video-id", "cli-video",
            "--synthetic-player-id", "cli-player",
            "--output-dir", str(tmp_path / "out"),
            "--name", "preds",
        ]
    )
    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


def test_evaluate_command_scores_predictions_against_synthetic_ground_truth(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    predictions_dir = tmp_path / "predictions"
    run_exit_code = main(
        [
            "run",
            "--detector", "majority",
            "--synthetic-seed", "1",
            "--synthetic-video-id", "cli-video",
            "--synthetic-player-id", "cli-player",
            "--output-dir", str(predictions_dir),
            "--name", "preds",
        ]
    )
    assert run_exit_code == 0
    capsys.readouterr()  # discard `run`'s stdout

    exit_code = main(
        [
            "evaluate",
            "--predictions", str(predictions_dir / "preds.parquet"),
            "--synthetic-seed", "1",
            "--synthetic-video-id", "cli-video",
            "--synthetic-player-id", "cli-player",
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {
        "event_precision_recall",
        "per_class_precision_recall",
        "macro_f1",
        "temporal_map",
        "boundary_error",
        "outcome_f1",
        "calibration",
        "confusion_matrix",
    }


def test_evaluate_command_requires_either_record_or_synthetic_args(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    predictions_dir = tmp_path / "predictions"
    main(
        [
            "run",
            "--detector", "majority",
            "--synthetic-seed", "1",
            "--synthetic-video-id", "cli-video",
            "--synthetic-player-id", "cli-player",
            "--output-dir", str(predictions_dir),
            "--name", "preds",
        ]
    )
    capsys.readouterr()

    exit_code = main(["evaluate", "--predictions", str(predictions_dir / "preds.parquet")])
    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


def test_compare_baselines_command_reports_every_baseline_and_ablation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "compare-baselines",
            "--seed", "0",
            "--num-players", "4",
            "--clips-per-player", "1",
            "--num-events-per-clip", "2",
            "--max-epochs", "1",
            "--hidden-channels", "4",
            "--num-blocks", "1",
            "--split", "val",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    reports = json.loads(captured.out)
    assert set(reports) == {"majority", "rules", "skeleton", "trajectory", "fused"}
    assert "not a measurement of real-world" in captured.err


def test_compare_modalities_command_reports_kinematics_only_and_multimodal_fused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "compare-modalities",
            "--seed", "0",
            "--num-players", "4",
            "--clips-per-player", "1",
            "--num-events-per-clip", "2",
            "--max-epochs", "1",
            "--hidden-channels", "4",
            "--num-blocks", "1",
            "--split", "val",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    reports = json.loads(captured.out)
    assert set(reports) == {"kinematics_only", "multimodal_fused"}
    assert "must not be read as evidence" in captured.err


def test_build_parser_registers_all_subcommands() -> None:
    from yoyovision_ml.events.cli import build_parser

    parser = build_parser()
    subparsers_action = next(
        action for action in parser._actions if action.dest == "command"
    )
    assert subparsers_action.choices is not None
    assert set(subparsers_action.choices) == {
        "train",
        "run",
        "evaluate",
        "compare-baselines",
        "compare-modalities",
    }
