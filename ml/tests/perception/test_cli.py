from __future__ import annotations

import json
from pathlib import Path

import pytest

from yoyovision_ml.dataset.io import save_record
from yoyovision_ml.dataset.schema import (
    DatasetRecord,
    DatasetVideo,
    NormalizedPoint,
    YoyoFrameAnnotation,
)
from yoyovision_ml.perception.cli import main


def _fake_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake mp4 bytes for CLI smoke test")
    return video_path


def _dataset_record_path(tmp_path: Path, duration_ms: int = 1000) -> Path:
    video = DatasetVideo(
        video_id="v1",
        player_id="p1",
        relative_path="videos/v1.mp4",
        checksum_sha256="a" * 64,
        duration_ms=duration_ms,
        width=1920,
        height=1080,
        source_fps=30.0,
    )
    record = DatasetRecord(
        record_id="r1",
        video=video,
        annotator_id="alex",
        ontology_version="dataset-ontology-v1",
        yoyo_track=[
            YoyoFrameAnnotation(
                frame_ms=ms, point=NormalizedPoint(x=0.5, y=0.5), visibility="visible"
            )
            for ms in range(0, duration_ms, 100)
        ],
    )
    return save_record(tmp_path, record)


def test_run_command_writes_parquet_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video_path = _fake_video(tmp_path)
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "run",
            str(video_path),
            "--duration-ms",
            "1000",
            "--fps",
            "30",
            "--output-dir",
            str(output_dir),
            "--name",
            "clip",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "clip.parquet").exists()
    assert (output_dir / "clip.json").exists()
    captured = capsys.readouterr()
    assert "Wrote feature table" in captured.out
    assert "Wrote metadata" in captured.out


def test_run_command_with_kalman_tracker(tmp_path: Path) -> None:
    video_path = _fake_video(tmp_path)
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "run",
            str(video_path),
            "--duration-ms",
            "1000",
            "--fps",
            "30",
            "--tracker-adapter",
            "kalman",
            "--tracker-max-gap-ms",
            "200",
            "--static-camera",
            "--output-dir",
            str(output_dir),
            "--name",
            "clip",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "clip.json").exists()
    metadata = json.loads((output_dir / "clip.json").read_text())
    assert metadata["model_versions"]["tracker"].startswith("kalman-yoyo-tracker@")


def test_run_command_reports_error_for_unconfigured_pytorch_weights(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video_path = _fake_video(tmp_path)

    exit_code = main(
        [
            "run",
            str(video_path),
            "--duration-ms",
            "1000",
            "--fps",
            "30",
            "--yoyo-adapter",
            "pytorch",
            "--output-dir",
            str(tmp_path / "out"),
            "--name",
            "clip",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err


def test_evaluate_command_prints_metrics_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video_path = _fake_video(tmp_path)
    record_path = _dataset_record_path(tmp_path, duration_ms=1000)

    exit_code = main(
        ["evaluate", str(video_path), str(record_path), "--duration-ms", "1000", "--fps", "30"]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["video"] == "clip.mp4"
    assert "detector_precision_recall" in report
    assert "track_coverage" in report
    assert "interpolation_rate" in report
    assert "normalized_centre_error" in report
    assert "centre_point_pixel_error" in report


def test_overlay_command_reports_error_without_cv2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video_path = _fake_video(tmp_path)
    output_path = tmp_path / "overlay.mp4"

    exit_code = main(
        [
            "overlay",
            str(video_path),
            "--duration-ms",
            "1000",
            "--fps",
            "30",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "cv2" in captured.err


def test_missing_command_argument_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        main([])
