"""Dataset-wide validation: the checks required by Prompt A requirement 9.

Runs across a whole loaded dataset (manifest + records), not just per-record
Pydantic field validation (which `schema.py` already handles at parse time --
e.g. malformed intervals or out-of-range coordinates fail before a record
even reaches this module). This module catches *cross-record* / *dataset*
level problems: unknown ontology labels, events outside the video they
belong to, duplicate IDs across the whole corpus, missing video files on
disk, split leakage, and same-family event overlap the ontology doesn't
explicitly allow.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from yoyovision_ml.dataset.ontology import EventOntology
from yoyovision_ml.dataset.schema import (
    DatasetManifest,
    DatasetRecord,
    DatasetVideo,
    TrickEventAnnotation,
)
from yoyovision_ml.domain import EventFamily


class IssueSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class IssueRule(StrEnum):
    INVALID_INTERVAL = "invalid_interval"
    OVERLAPPING_INCOMPATIBLE_LABELS = "overlapping_incompatible_labels"
    EVENT_OUTSIDE_VIDEO_DURATION = "event_outside_video_duration"
    UNKNOWN_ONTOLOGY_LABEL = "unknown_ontology_label"
    DUPLICATE_ID = "duplicate_id"
    MISSING_FILE = "missing_file"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    LEAKED_PLAYER_ACROSS_SPLITS = "leaked_player_across_splits"


@dataclass(slots=True, frozen=True)
class ValidationIssue:
    rule: IssueRule
    severity: IssueSeverity
    message: str
    record_id: str | None = None
    video_id: str | None = None


@dataclass(slots=True)
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    @property
    def is_valid(self) -> bool:
        """A dataset is valid if it has zero ERROR-severity issues.
        WARNINGs (e.g. ontology-permitted overlaps flagged for awareness,
        or checksum drift on files not yet re-hashed) do not block use."""
        return len(self.errors) == 0

    def add(
        self,
        rule: IssueRule,
        severity: IssueSeverity,
        message: str,
        *,
        record_id: str | None = None,
        video_id: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                rule=rule,
                severity=severity,
                message=message,
                record_id=record_id,
                video_id=video_id,
            )
        )


def _check_invalid_intervals(record: DatasetRecord, report: ValidationReport) -> None:
    # `schema.py`'s model_validator already rejects end_ms <= start_ms at
    # parse time; this re-checks defensively for records constructed via
    # `model_construct()` or loaded from a schema_version this validator
    # doesn't fully trust, and additionally flags start_ms beyond the video.
    for event in record.trick_events:
        if event.end_ms <= event.start_ms:
            report.add(
                IssueRule.INVALID_INTERVAL,
                IssueSeverity.ERROR,
                f"Event '{event.event_id}' has end_ms <= start_ms "
                f"({event.end_ms} <= {event.start_ms}).",
                record_id=record.record_id,
                video_id=record.video.video_id,
            )


def _check_events_outside_video_duration(record: DatasetRecord, report: ValidationReport) -> None:
    duration_ms = record.video.duration_ms
    for event in record.trick_events:
        if event.start_ms > duration_ms or event.end_ms > duration_ms:
            report.add(
                IssueRule.EVENT_OUTSIDE_VIDEO_DURATION,
                IssueSeverity.ERROR,
                f"Event '{event.event_id}' ({event.start_ms}-{event.end_ms}ms) exceeds "
                f"video duration ({duration_ms}ms).",
                record_id=record.record_id,
                video_id=record.video.video_id,
            )
    for deduction in record.deductions:
        if deduction.timestamp_ms > duration_ms:
            report.add(
                IssueRule.EVENT_OUTSIDE_VIDEO_DURATION,
                IssueSeverity.ERROR,
                f"Deduction '{deduction.deduction_id}' at {deduction.timestamp_ms}ms exceeds "
                f"video duration ({duration_ms}ms).",
                record_id=record.record_id,
                video_id=record.video.video_id,
            )
    for click in record.judge_clicks:
        if click.timestamp_ms > duration_ms:
            report.add(
                IssueRule.EVENT_OUTSIDE_VIDEO_DURATION,
                IssueSeverity.ERROR,
                f"Judge click '{click.click_id}' at {click.timestamp_ms}ms exceeds "
                f"video duration ({duration_ms}ms).",
                record_id=record.record_id,
                video_id=record.video.video_id,
            )


def _check_overlapping_incompatible_labels(
    record: DatasetRecord, ontology: EventOntology, report: ValidationReport
) -> None:
    by_family: dict[EventFamily, list[TrickEventAnnotation]] = defaultdict(list)
    for event in record.trick_events:
        by_family[event.family].append(event)

    for family, events in by_family.items():
        if ontology.allows_overlap(family):
            continue
        ordered = sorted(events, key=lambda e: e.start_ms)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if later.start_ms < earlier.end_ms:
                report.add(
                    IssueRule.OVERLAPPING_INCOMPATIBLE_LABELS,
                    IssueSeverity.ERROR,
                    f"Events '{earlier.event_id}' ({earlier.start_ms}-{earlier.end_ms}ms) and "
                    f"'{later.event_id}' ({later.start_ms}-{later.end_ms}ms) are both family "
                    f"'{family}', which does not permit overlap in ontology "
                    f"'{ontology.version}'.",
                    record_id=record.record_id,
                    video_id=record.video.video_id,
                )


def _check_unknown_ontology_labels(
    record: DatasetRecord, ontology: EventOntology, report: ValidationReport
) -> None:
    for event in record.trick_events:
        if not ontology.is_known_label(event.label):
            report.add(
                IssueRule.UNKNOWN_ONTOLOGY_LABEL,
                IssueSeverity.ERROR,
                f"Event '{event.event_id}' uses label '{event.label}', which is not in "
                f"ontology '{ontology.version}'.",
                record_id=record.record_id,
                video_id=record.video.video_id,
            )
        else:
            expected_family = ontology.family_for_label(event.label)
            if expected_family is not None and expected_family != event.family:
                report.add(
                    IssueRule.UNKNOWN_ONTOLOGY_LABEL,
                    IssueSeverity.ERROR,
                    f"Event '{event.event_id}' has label '{event.label}' (ontology family "
                    f"'{expected_family}') but is tagged family '{event.family}'.",
                    record_id=record.record_id,
                    video_id=record.video.video_id,
                )


def _check_missing_files(
    videos: dict[str, DatasetVideo], dataset_dir: Path, report: ValidationReport
) -> None:
    for video in videos.values():
        full_path = dataset_dir / video.relative_path
        if not full_path.exists():
            report.add(
                IssueRule.MISSING_FILE,
                IssueSeverity.ERROR,
                f"Video file for '{video.video_id}' not found at '{video.relative_path}'.",
                video_id=video.video_id,
            )
            continue
        actual_checksum = hashlib.sha256(full_path.read_bytes()).hexdigest()
        if actual_checksum != video.checksum_sha256:
            report.add(
                IssueRule.CHECKSUM_MISMATCH,
                IssueSeverity.WARNING,
                f"Video '{video.video_id}' on-disk checksum ({actual_checksum}) does not "
                f"match manifest checksum ({video.checksum_sha256}).",
                video_id=video.video_id,
            )


def _check_duplicate_ids(
    manifest: DatasetManifest, records: list[DatasetRecord], report: ValidationReport
) -> None:
    record_ids = [r.record_id for r in records]
    for dup in {i for i in record_ids if record_ids.count(i) > 1}:
        report.add(
            IssueRule.DUPLICATE_ID,
            IssueSeverity.ERROR,
            f"Duplicate record_id '{dup}' across the dataset.",
        )

    all_event_ids: list[str] = []
    all_deduction_ids: list[str] = []
    for record in records:
        all_event_ids.extend(e.event_id for e in record.trick_events)
        all_deduction_ids.extend(d.deduction_id for d in record.deductions)
    for dup in {i for i in all_event_ids if all_event_ids.count(i) > 1}:
        report.add(
            IssueRule.DUPLICATE_ID,
            IssueSeverity.ERROR,
            f"Duplicate event_id '{dup}' across the dataset (event IDs must be globally unique).",
        )
    for dup in {i for i in all_deduction_ids if all_deduction_ids.count(i) > 1}:
        report.add(
            IssueRule.DUPLICATE_ID,
            IssueSeverity.ERROR,
            f"Duplicate deduction_id '{dup}' across the dataset.",
        )


def _check_leaked_players_across_splits(
    manifest: DatasetManifest, videos: dict[str, DatasetVideo], report: ValidationReport
) -> None:
    if not manifest.splits:
        return
    player_splits: dict[str, set[str]] = defaultdict(set)
    for video_id, split in manifest.splits.items():
        video = videos.get(video_id)
        if video is None:
            continue
        player_splits[video.player_id].add(str(split))

    for player_id, splits in player_splits.items():
        if len(splits) > 1:
            report.add(
                IssueRule.LEAKED_PLAYER_ACROSS_SPLITS,
                IssueSeverity.ERROR,
                f"Player '{player_id}' appears in multiple splits: {sorted(splits)}.",
            )


def validate_dataset(
    manifest: DatasetManifest,
    records: list[DatasetRecord],
    dataset_dir: Path,
    ontology: EventOntology,
) -> ValidationReport:
    """Runs every dataset-level validation rule and returns a full report
    (never raises on validation findings -- only a report consumer decides
    whether `report.is_valid` should block anything)."""
    report = ValidationReport()
    videos = {r.video.video_id: r.video for r in records}

    for record in records:
        _check_invalid_intervals(record, report)
        _check_events_outside_video_duration(record, report)
        _check_overlapping_incompatible_labels(record, ontology, report)
        _check_unknown_ontology_labels(record, ontology, report)

    _check_missing_files(videos, dataset_dir, report)
    _check_duplicate_ids(manifest, records, report)
    _check_leaked_players_across_splits(manifest, videos, report)

    return report
