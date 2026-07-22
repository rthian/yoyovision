"""Prompt D "CALIBRATION" section: "Implement scripts to compare model
output with expert judges using: mean absolute error, Spearman rank
correlation, Pearson correlation, intraclass correlation where appropriate,
event-count precision and recall, Bland-Altman-style error summaries, score
calibration plots."

Distinct from `yoyovision_ml.events.calibration`, which calibrates the
Prompt C temporal-event-detector's own confidence *temperature* -- this
module calibrates model *score/category* output against expert human
judges. Pure numpy (no scipy dependency in this package): `_rank_data`
reimplements tie-aware ranking, and `intraclass_correlation` reimplements
ICC(3,1) (Shrout & Fleiss, 1979) from scratch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from yoyovision_ml.domain import AnalysisEventPrediction
from yoyovision_ml.perception.errors import MissingOptionalDependencyError
from yoyovision_ml.scoring.judges import match_clicks_to_events
from yoyovision_ml.scoring.types import JudgeClick


def _rank_data(values: np.ndarray) -> np.ndarray:
    """Average ranks, 1-indexed, tie-aware -- mirrors `scipy.stats.rankdata`'s
    default `method="average"` without depending on scipy."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = average_rank
        i = j + 1
    return ranks


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """None when undefined (fewer than 2 points, or either series constant)
    -- never a fabricated correlation."""
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2:
        return None
    return pearson_correlation(_rank_data(a), _rank_data(b))


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


@dataclass(slots=True, frozen=True)
class BlandAltmanSummary:
    """Mean bias + 95% limits of agreement between two paired measurement
    series (Bland & Altman, 1986)."""

    mean_diff: float
    std_diff: float
    lower_loa: float
    upper_loa: float
    mean_of_means: float


def bland_altman_summary(a: np.ndarray, b: np.ndarray) -> BlandAltmanSummary:
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return BlandAltmanSummary(
        mean_diff=mean_diff,
        std_diff=std_diff,
        lower_loa=mean_diff - 1.96 * std_diff,
        upper_loa=mean_diff + 1.96 * std_diff,
        mean_of_means=float(np.mean((a + b) / 2.0)),
    )


def intraclass_correlation(ratings: np.ndarray) -> float | None:
    """ICC(3,1): two-way mixed-effects, single measurement, consistency
    (Shrout & Fleiss, 1979) -- appropriate when the *same fixed set* of
    raters (e.g. "the model" and "judge X") rates every subject, which is
    Prompt D's use case ("intraclass correlation where appropriate").
    `ratings` is `(n_subjects, n_raters)`; rows containing any NaN are
    dropped. Returns None when fewer than 2 valid subjects or raters remain,
    or the between-subject variance is degenerate."""
    ratings = np.asarray(ratings, dtype=float)
    if ratings.ndim != 2:
        raise ValueError("ratings must be a 2D (n_subjects, n_raters) array")
    ratings = ratings[~np.isnan(ratings).any(axis=1)]
    n, k = ratings.shape
    if n < 2 or k < 2:
        return None

    grand_mean = ratings.mean()
    subject_means = ratings.mean(axis=1)
    rater_means = ratings.mean(axis=0)

    ss_total = float(np.sum((ratings - grand_mean) ** 2))
    ss_subjects = float(k * np.sum((subject_means - grand_mean) ** 2))
    ss_raters = float(n * np.sum((rater_means - grand_mean) ** 2))
    ss_error = ss_total - ss_subjects - ss_raters

    ms_subjects = ss_subjects / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0.0
    denominator = ms_subjects + (k - 1) * ms_error
    if denominator == 0:
        return None
    return float((ms_subjects - ms_error) / denominator)


@dataclass(slots=True, frozen=True)
class PairedAgreement:
    """Prompt D's core statistics between two paired arrays of scores
    (model vs one judge, judge vs judge, or model vs judge-mean)."""

    n: int
    mean_absolute_error: float
    pearson_r: float | None
    spearman_rho: float | None
    bland_altman: BlandAltmanSummary


def paired_agreement(a: Sequence[float], b: Sequence[float]) -> PairedAgreement:
    arr_a, arr_b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(arr_a) != len(arr_b):
        raise ValueError(f"paired arrays must be equal length, got {len(arr_a)} vs {len(arr_b)}")
    if len(arr_a) == 0:
        raise ValueError("paired arrays must be non-empty")
    return PairedAgreement(
        n=len(arr_a),
        mean_absolute_error=mean_absolute_error(arr_a, arr_b),
        pearson_r=pearson_correlation(arr_a, arr_b),
        spearman_rho=spearman_correlation(arr_a, arr_b),
        bland_altman=bland_altman_summary(arr_a, arr_b),
    )


@dataclass(slots=True, frozen=True)
class EventCountAgreement:
    """Prompt D: "event-count precision and recall." Judge clicks are
    treated as the reference/ground truth: precision = share of model
    events matched by some click within `tolerance_ms`; recall = share of
    judge clicks matched by some model event."""

    model_event_count: int
    judge_click_count: int
    matched_event_count: int
    matched_click_count: int
    precision: float
    recall: float
    mean_boundary_error_ms: float | None


def event_count_agreement(
    events: Sequence[AnalysisEventPrediction],
    clicks: Sequence[JudgeClick],
    tolerance_ms: int = 1000,
) -> EventCountAgreement:
    click_matches = match_clicks_to_events(clicks, events, tolerance_ms=tolerance_ms)
    matched_click_count = sum(1 for m in click_matches if m.matched_event_label is not None)

    matched_event_count = 0
    boundary_errors: list[int] = []
    for event in events:
        candidate_clicks = [
            c
            for c in clicks
            if abs(event.start_ms - c.timestamp_ms) <= tolerance_ms
            and (c.associated_label is None or c.associated_label == event.label)
        ]
        if not candidate_clicks:
            continue
        matched_event_count += 1
        best = min(candidate_clicks, key=lambda c: abs(event.start_ms - c.timestamp_ms))
        boundary_errors.append(abs(event.start_ms - best.timestamp_ms))

    precision = matched_event_count / len(events) if events else 0.0
    recall = matched_click_count / len(clicks) if clicks else 0.0
    mean_error = float(np.mean(boundary_errors)) if boundary_errors else None

    return EventCountAgreement(
        model_event_count=len(events),
        judge_click_count=len(clicks),
        matched_event_count=matched_event_count,
        matched_click_count=matched_click_count,
        precision=round(precision, 4),
        recall=round(recall, 4),
        mean_boundary_error_ms=round(mean_error, 1) if mean_error is not None else None,
    )


def render_calibration_plot(
    model_scores: Sequence[float],
    judge_scores: Sequence[float],
    output_path: Path,
    title: str = "Model vs judge score calibration",
) -> Path:
    """Prompt D: "score calibration plots" -- a scatter of model vs judge
    score with a 45-degree perfect-agreement reference line. Lazily imports
    matplotlib (an optional `yoyovision-ml[plotting]` dependency, like
    torch/mediapipe elsewhere in this package) so nothing in the core
    scoring/calibration path *requires* a plotting backend to be installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingOptionalDependencyError("matplotlib", "plotting") from exc

    if len(model_scores) != len(judge_scores):
        raise ValueError("model_scores and judge_scores must be paired (equal length)")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(judge_scores, model_scores, alpha=0.7)
    combined = [*judge_scores, *model_scores]
    lo, hi = (min(combined), max(combined)) if combined else (0.0, 1.0)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="perfect agreement")
    ax.set_xlabel("Judge score")
    ax.set_ylabel("Model score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def paired_agreement_to_dict(agreement: PairedAgreement) -> dict[str, object]:
    """JSON-friendly serialization for CLI/report output."""
    payload = asdict(agreement)
    return payload
