"""Human-readable analysis report generation.

Prompt F: "Generate a human-readable analysis report." Complements
`exports.py`'s JSON/CSV machine-readable exports with a Markdown summary a
reviewer (or judge) can read directly -- model versions, device/runtime,
per-stage timing, event/score summary, and monitoring signals, with the same
"never expose a local filesystem path" discipline as `PerceptionMetadata`
(only a basename is accepted for `video_filename`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from yoyovision_ml.domain import ScoreBreakdown
from yoyovision_ml.inference.device import DeviceInfo
from yoyovision_ml.inference.monitoring import MonitoringSignals
from yoyovision_ml.inference.timing import StageTimings

if TYPE_CHECKING:
    # Deferred: `pipeline.py` imports this package for `StageTimings`/
    # `CancellationToken`/etc., so importing `PipelineResult` at module load
    # time here would be circular. Only used for type hints.
    from yoyovision_ml.pipeline import PipelineResult


def generate_human_readable_report(
    *,
    job_id: str,
    video_filename: str,
    pipeline_version: str,
    result: PipelineResult,
    timings: StageTimings,
    device_info: DeviceInfo,
    runtime_versions: dict[str, str],
    monitoring: MonitoringSignals | None = None,
) -> str:
    """Returns a Markdown report string. `video_filename` must be a basename
    (never an absolute/local path) -- callers pass `Path(...).name`, matching
    the convention `perception.artifact.PerceptionMetadata` already uses."""
    lines: list[str] = [
        f"# YoYoVision Analysis Report — job `{job_id}`",
        "",
        "_Training and judge-assistance tool only. This report is not a "
        "certified score from IYYF, WYYC, or any competition body._",
        "",
        "## Summary",
        "",
        f"- **Video**: `{video_filename}`",
        f"- **Pipeline version**: `{pipeline_version}`",
        f"- **Device**: `{device_info.resolved}` (requested `{device_info.requested}`"
        f"{'' if device_info.available else ', unavailable — fell back'})",
        "",
        "## Model versions",
        "",
    ]
    for adapter_name, version in sorted(result.model_versions.items()):
        lines.append(f"- **{adapter_name}**: `{version}`")

    lines += ["", "## Runtime", ""]
    for key, value in sorted(runtime_versions.items()):
        lines.append(f"- **{key}**: `{value}`")

    lines += ["", "## Stage durations", "", "| Stage | Duration (ms) |", "| --- | --- |"]
    for stage_name, duration_ms in timings.durations_ms.items():
        lines.append(f"| {stage_name} | {duration_ms:.1f} |")
    lines.append(f"| **total** | **{timings.total_ms:.1f}** |")

    lines += _score_section(result.score)
    lines += _events_section(result)

    if monitoring is not None:
        lines += _monitoring_section(monitoring)

    if result.score.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {warning}" for warning in result.score.warnings]

    return "\n".join(lines) + "\n"


def _score_section(score: ScoreBreakdown) -> list[str]:
    return [
        "",
        "## Score",
        "",
        f"- **Final score**: {score.final_score:.2f} (confidence {score.confidence:.2f})",
        f"- **Technical**: raw {score.technical_raw:.2f}, scaled {score.technical_scaled:.2f}",
        f"- **Freestyle Evaluation**: raw {score.freestyle_evaluation_raw:.2f}, "
        f"scaled {score.freestyle_evaluation_scaled:.2f}",
        f"- **Major deductions**: -{score.major_deductions:.2f}",
        f"- **Ruleset**: `{score.ruleset_version}`",
    ]


def _events_section(result: PipelineResult) -> list[str]:
    lines = ["", "## Events", "", f"- **Total events detected**: {len(result.events)}"]
    if result.events:
        by_family: dict[str, int] = {}
        for event in result.events:
            by_family[event.family.value] = by_family.get(event.family.value, 0) + 1
        lines.append("")
        lines.append("| Family | Count |")
        lines.append("| --- | --- |")
        for family, count in sorted(by_family.items()):
            lines.append(f"| {family} | {count} |")
    lines.append(f"- **Major deductions**: {len(result.deductions)}")
    return lines


def _monitoring_section(monitoring: MonitoringSignals) -> list[str]:
    lines = [
        "",
        "## Monitoring signals",
        "",
        f"- **Average event confidence**: {monitoring.avg_confidence:.3f}",
        f"- **Low-confidence event rate**: {monitoring.low_confidence_rate:.1%}",
        f"- **Failed-track rate**: {monitoring.failed_track_rate:.1%}",
    ]
    if monitoring.class_drift_score is not None:
        lines.append(f"- **Class drift score** (vs. reference): {monitoring.class_drift_score:.3f}")
    if monitoring.confidence_drift_score is not None:
        lines.append(
            f"- **Confidence drift score** (vs. reference): {monitoring.confidence_drift_score:.3f}"
        )
    return lines
