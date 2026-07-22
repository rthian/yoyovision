"""Prompt D "CALIBRATION" section: mean absolute error, Spearman rank
correlation, Pearson correlation, intraclass correlation, event-count
precision/recall, Bland-Altman-style error summaries, score calibration
plots. Tests `scoring.calibration`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import make_event_prediction, make_judge_click

from yoyovision_ml.perception.errors import MissingOptionalDependencyError
from yoyovision_ml.scoring.calibration import (
    bland_altman_summary,
    event_count_agreement,
    intraclass_correlation,
    mean_absolute_error,
    paired_agreement,
    paired_agreement_to_dict,
    pearson_correlation,
    render_calibration_plot,
    spearman_correlation,
)


def test_mean_absolute_error_basic() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 4.0, 3.0])
    assert mean_absolute_error(a, b) == pytest.approx(2.0 / 3.0)


def test_pearson_correlation_perfect_linear_relationship() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 4.0, 6.0, 8.0])
    assert pearson_correlation(a, b) == pytest.approx(1.0)


def test_pearson_correlation_none_when_degenerate() -> None:
    assert pearson_correlation(np.array([1.0]), np.array([2.0])) is None
    assert pearson_correlation(np.array([1.0, 1.0]), np.array([1.0, 2.0])) is None
    assert pearson_correlation(np.array([1.0, 2.0]), np.array([5.0, 5.0])) is None


def test_spearman_correlation_perfect_monotonic_nonlinear_relationship() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 4.0, 9.0, 16.0])  # non-linear but monotonic
    assert spearman_correlation(a, b) == pytest.approx(1.0)


def test_spearman_correlation_none_with_fewer_than_two_points() -> None:
    assert spearman_correlation(np.array([1.0]), np.array([1.0])) is None


def test_spearman_correlation_handles_ties() -> None:
    a = np.array([1.0, 1.0, 2.0, 3.0])
    b = np.array([1.0, 1.0, 2.0, 3.0])
    assert spearman_correlation(a, b) == pytest.approx(1.0)


def test_bland_altman_summary_zero_bias_for_identical_series() -> None:
    a = np.array([5.0, 6.0, 7.0])
    b = np.array([5.0, 6.0, 7.0])
    summary = bland_altman_summary(a, b)
    assert summary.mean_diff == 0.0
    assert summary.std_diff == 0.0
    assert summary.lower_loa == 0.0
    assert summary.upper_loa == 0.0
    assert summary.mean_of_means == pytest.approx(6.0)


def test_bland_altman_summary_detects_constant_bias() -> None:
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([8.0, 9.0, 10.0])
    summary = bland_altman_summary(a, b)
    assert summary.mean_diff == pytest.approx(2.0)
    assert summary.std_diff == 0.0


def test_intraclass_correlation_perfect_agreement_across_raters() -> None:
    ratings = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    icc = intraclass_correlation(ratings)
    assert icc is not None
    assert icc == pytest.approx(1.0, abs=1e-6)


def test_intraclass_correlation_none_with_too_few_subjects() -> None:
    assert intraclass_correlation(np.array([[1.0, 2.0]])) is None


def test_intraclass_correlation_drops_rows_with_nan() -> None:
    ratings = np.array([[1.0, 1.0], [np.nan, 2.0], [3.0, 3.0], [4.0, 4.0]])
    icc = intraclass_correlation(ratings)
    assert icc is not None  # 3 valid subjects remain after dropping the NaN row


def test_intraclass_correlation_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        intraclass_correlation(np.array([1.0, 2.0, 3.0]))


def test_paired_agreement_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_agreement([1.0, 2.0], [1.0])


def test_paired_agreement_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        paired_agreement([], [])


def test_paired_agreement_bundles_all_required_statistics() -> None:
    agreement = paired_agreement([70.0, 80.0, 90.0], [72.0, 78.0, 91.0])
    assert agreement.n == 3
    assert agreement.mean_absolute_error >= 0.0
    assert agreement.pearson_r is not None
    assert agreement.spearman_rho is not None
    assert agreement.bland_altman is not None


def test_paired_agreement_to_dict_is_json_serializable() -> None:
    import json

    agreement = paired_agreement([70.0, 80.0], [72.0, 78.0])
    payload = paired_agreement_to_dict(agreement)
    json.dumps(payload)  # must not raise
    assert payload["n"] == 2
    assert "bland_altman" in payload


def test_event_count_agreement_perfect_match() -> None:
    events = [make_event_prediction(label="mount_1", start_ms=1000)]
    clicks = [make_judge_click(timestamp_ms=1000)]
    result = event_count_agreement(events, clicks, tolerance_ms=500)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.mean_boundary_error_ms == 0.0


def test_event_count_agreement_no_events_or_clicks() -> None:
    result = event_count_agreement([], [], tolerance_ms=500)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.mean_boundary_error_ms is None


def test_event_count_agreement_partial_match_computes_precision_and_recall() -> None:
    events = [
        make_event_prediction(label="mount_1", start_ms=1000),
        make_event_prediction(label="mount_2", start_ms=9000),  # unmatched by any click
    ]
    clicks = [
        make_judge_click(click_id="c1", timestamp_ms=1000),
        make_judge_click(click_id="c2", timestamp_ms=20000),  # unmatched by any event
    ]
    result = event_count_agreement(events, clicks, tolerance_ms=500)
    assert result.model_event_count == 2
    assert result.judge_click_count == 2
    assert result.matched_event_count == 1
    assert result.matched_click_count == 1
    assert result.precision == 0.5
    assert result.recall == 0.5


def test_render_calibration_plot_raises_missing_optional_dependency_when_matplotlib_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "matplotlib":
            raise ImportError("simulated missing matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(MissingOptionalDependencyError):
        render_calibration_plot([1.0], [1.0], tmp_path / "plot.png")


def test_render_calibration_plot_rejects_mismatched_lengths(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="paired"):
        render_calibration_plot([1.0, 2.0], [1.0], tmp_path / "plot.png")


def test_render_calibration_plot_writes_a_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    output_path = tmp_path / "calibration.png"
    result_path = render_calibration_plot([70.0, 80.0, 90.0], [72.0, 78.0, 91.0], output_path)
    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
