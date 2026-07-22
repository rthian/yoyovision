"""JSON and CSV export serialization for analysis results.

Pure serialization: takes already-persisted domain objects and produces
export payloads. Filename sanitization lives here too since it is tightly
coupled to what an export "is" (one export == one sanitized filename).
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from yoyovision_ml.domain import (
    AnalysisEvent,
    MajorDeduction,
    ScoreBreakdown,
    VideoAsset,
)

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SANITIZED_FILENAME_LENGTH = 128


def sanitize_export_filename(raw_name: str, extension: str) -> str:
    """Never trusts client-provided filenames; strips path separators and any
    character outside a safe allowlist, and always appends the intended
    extension server-side (never trusts a client-supplied extension)."""
    base = raw_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    base = base.rsplit(".", 1)[0] if "." in base else base
    safe = _UNSAFE_FILENAME_CHARS.sub("_", base).strip("._") or "export"
    safe = safe[:_MAX_SANITIZED_FILENAME_LENGTH]
    clean_extension = extension.lstrip(".")
    return f"{safe}.{clean_extension}"


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if hasattr(value, "value"):  # StrEnum members
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def export_analysis_json(
    video: VideoAsset,
    events: list[AnalysisEvent],
    deductions: list[MajorDeduction],
    score: ScoreBreakdown | None,
    pipeline_version: str,
    ruleset_version: str,
) -> str:
    payload = {
        "disclaimer": (
            "This report is an unofficial training/judge-assistance estimate. "
            "It is not certified by IYYF, WYYC, or any competition body."
        ),
        "pipeline_version": pipeline_version,
        "ruleset_version": ruleset_version,
        "video": asdict(video),
        "events": [asdict(e) for e in events],
        "major_deductions": [asdict(d) for d in deductions],
        "score": asdict(score) if score is not None else None,
    }
    return json.dumps(payload, indent=2, default=_json_default)


_EVENT_CSV_COLUMNS = [
    "id",
    "label",
    "family",
    "start_ms",
    "end_ms",
    "confidence",
    "outcome",
    "difficulty_band",
    "source",
    "review_status",
    "model_name",
    "model_version",
]


def export_events_csv(events: list[AnalysisEvent]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_EVENT_CSV_COLUMNS)
    writer.writeheader()
    for event in sorted(events, key=lambda e: e.start_ms):
        row = {column: getattr(event, column) for column in _EVENT_CSV_COLUMNS}
        for key, value in row.items():
            if hasattr(value, "value"):
                row[key] = value.value
        writer.writerow(row)
    return buffer.getvalue()


_DEDUCTION_CSV_COLUMNS = [
    "id",
    "type",
    "timestamp_ms",
    "quantity",
    "points",
    "confidence",
    "source",
    "review_status",
]


def export_deductions_csv(deductions: list[MajorDeduction]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_DEDUCTION_CSV_COLUMNS)
    writer.writeheader()
    for deduction in sorted(deductions, key=lambda d: d.timestamp_ms):
        row = {column: getattr(deduction, column) for column in _DEDUCTION_CSV_COLUMNS}
        for key, value in row.items():
            if hasattr(value, "value"):
                row[key] = value.value
        writer.writerow(row)
    return buffer.getvalue()
