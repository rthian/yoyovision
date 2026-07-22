from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from yoyovision_ml.dataset.ontology import default_ontology
from yoyovision_ml.dataset.schema import (
    AnnotationProvenance,
    DatasetManifest,
    DatasetRecord,
    DatasetVideo,
    SplitName,
    TrickEventAnnotation,
)
from yoyovision_ml.dataset.validators import IssueRule, IssueSeverity, validate_dataset
from yoyovision_ml.domain import EventFamily, Outcome, Source

NOW = datetime(2026, 1, 1, tzinfo=UTC)
ONTOLOGY = default_ontology()


def _provenance() -> AnnotationProvenance:
    return AnnotationProvenance(annotator_id="alex", source=Source.HUMAN, annotated_at=NOW)


def _video(video_id: str = "v1", player_id: str = "p1", duration_ms: int = 10_000) -> DatasetVideo:
    return DatasetVideo(
        video_id=video_id,
        player_id=player_id,
        relative_path=f"videos/{video_id}.mp4",
        checksum_sha256="a" * 64,
        duration_ms=duration_ms,
        width=1920,
        height=1080,
        source_fps=30.0,
    )


def _event(
    event_id: str, family: EventFamily, start_ms: int, end_ms: int, label: str
) -> TrickEventAnnotation:
    return TrickEventAnnotation(
        event_id=event_id,
        label=label,
        family=family,
        start_ms=start_ms,
        end_ms=end_ms,
        outcome=Outcome.SUCCESS,
        provenance=_provenance(),
    )


def _manifest(
    record_paths: list[str], splits: dict[str, SplitName] | None = None
) -> DatasetManifest:
    return DatasetManifest(
        dataset_version="test",
        ontology_version=ONTOLOGY.version,
        created_at=NOW,
        video_ids=[],
        record_paths=record_paths,
        splits=splits,
    )


def _write_placeholder_video(dataset_dir: Path, video: DatasetVideo) -> None:
    path = dataset_dir / video.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


def test_valid_dataset_has_no_errors(tmp_path: Path) -> None:
    video = _video()
    _write_placeholder_video(tmp_path, video)
    actual_checksum = hashlib.sha256((tmp_path / video.relative_path).read_bytes()).hexdigest()
    video = video.model_copy(update={"checksum_sha256": actual_checksum})
    record = DatasetRecord(
        record_id="r1",
        video=video,
        annotator_id="alex",
        ontology_version=ONTOLOGY.version,
        trick_events=[_event("e1", EventFamily.MOUNT, 0, 500, "basic_mount")],
    )
    report = validate_dataset(_manifest(["r1"]), [record], tmp_path, ONTOLOGY)
    assert report.is_valid
    assert report.errors == []


def test_detects_events_outside_video_duration(tmp_path: Path) -> None:
    video = _video(duration_ms=1000)
    _write_placeholder_video(tmp_path, video)
    record = DatasetRecord(
        record_id="r1",
        video=video,
        annotator_id="alex",
        ontology_version=ONTOLOGY.version,
        trick_events=[_event("e1", EventFamily.MOUNT, 500, 2000, "basic_mount")],
    )
    report = validate_dataset(_manifest(["r1"]), [record], tmp_path, ONTOLOGY)
    assert not report.is_valid
    assert any(i.rule == IssueRule.EVENT_OUTSIDE_VIDEO_DURATION for i in report.errors)


def test_detects_unknown_ontology_label(tmp_path: Path) -> None:
    video = _video()
    _write_placeholder_video(tmp_path, video)
    record = DatasetRecord(
        record_id="r1",
        video=video,
        annotator_id="alex",
        ontology_version=ONTOLOGY.version,
        trick_events=[_event("e1", EventFamily.MOUNT, 0, 500, "totally_made_up_trick")],
    )
    report = validate_dataset(_manifest(["r1"]), [record], tmp_path, ONTOLOGY)
    assert not report.is_valid
    assert any(i.rule == IssueRule.UNKNOWN_ONTOLOGY_LABEL for i in report.errors)


def test_detects_overlapping_incompatible_same_family_events(tmp_path: Path) -> None:
    video = _video()
    _write_placeholder_video(tmp_path, video)
    record = DatasetRecord(
        record_id="r1",
        video=video,
        annotator_id="alex",
        ontology_version=ONTOLOGY.version,
        trick_events=[
            _event("e1", EventFamily.MOUNT, 0, 1000, "basic_mount"),
            _event("e2", EventFamily.MOUNT, 500, 1500, "brain_twister_mount"),
        ],
    )
    report = validate_dataset(_manifest(["r1"]), [record], tmp_path, ONTOLOGY)
    assert not report.is_valid
    assert any(i.rule == IssueRule.OVERLAPPING_INCOMPATIBLE_LABELS for i in report.errors)


def test_allows_overlap_for_families_ontology_permits(tmp_path: Path) -> None:
    video = _video()
    _write_placeholder_video(tmp_path, video)
    record = DatasetRecord(
        record_id="r1",
        video=video,
        annotator_id="alex",
        ontology_version=ONTOLOGY.version,
        trick_events=[
            _event("e1", EventFamily.BODY_TRICK, 0, 1000, "around_the_body"),
            _event("e2", EventFamily.BODY_TRICK, 500, 1500, "around_the_body"),
        ],
    )
    report = validate_dataset(_manifest(["r1"]), [record], tmp_path, ONTOLOGY)
    assert report.is_valid


def test_detects_missing_video_file(tmp_path: Path) -> None:
    video = _video()  # never written to disk
    record = DatasetRecord(
        record_id="r1",
        video=video,
        annotator_id="alex",
        ontology_version=ONTOLOGY.version,
    )
    report = validate_dataset(_manifest(["r1"]), [record], tmp_path, ONTOLOGY)
    assert not report.is_valid
    assert any(i.rule == IssueRule.MISSING_FILE for i in report.errors)


def test_detects_duplicate_record_ids(tmp_path: Path) -> None:
    video = _video()
    _write_placeholder_video(tmp_path, video)
    record_a = DatasetRecord(
        record_id="dup",
        video=video,
        annotator_id="alex",
        ontology_version=ONTOLOGY.version,
    )
    record_b = DatasetRecord(
        record_id="dup",
        video=video,
        annotator_id="bo",
        ontology_version=ONTOLOGY.version,
    )
    report = validate_dataset(_manifest(["r1", "r2"]), [record_a, record_b], tmp_path, ONTOLOGY)
    assert not report.is_valid
    assert any(i.rule == IssueRule.DUPLICATE_ID for i in report.errors)


def test_detects_duplicate_event_ids_across_records(tmp_path: Path) -> None:
    video_a = _video(video_id="va")
    video_b = _video(video_id="vb")
    _write_placeholder_video(tmp_path, video_a)
    _write_placeholder_video(tmp_path, video_b)
    record_a = DatasetRecord(
        record_id="ra",
        video=video_a,
        annotator_id="alex",
        ontology_version=ONTOLOGY.version,
        trick_events=[_event("shared_id", EventFamily.MOUNT, 0, 500, "basic_mount")],
    )
    record_b = DatasetRecord(
        record_id="rb",
        video=video_b,
        annotator_id="bo",
        ontology_version=ONTOLOGY.version,
        trick_events=[_event("shared_id", EventFamily.HOP, 0, 500, "eli_hop")],
    )
    report = validate_dataset(_manifest(["ra", "rb"]), [record_a, record_b], tmp_path, ONTOLOGY)
    assert not report.is_valid
    assert any(i.rule == IssueRule.DUPLICATE_ID for i in report.errors)


def test_detects_leaked_player_across_splits(tmp_path: Path) -> None:
    video_a = _video(video_id="va", player_id="shared_player")
    video_b = _video(video_id="vb", player_id="shared_player")
    _write_placeholder_video(tmp_path, video_a)
    _write_placeholder_video(tmp_path, video_b)
    record_a = DatasetRecord(
        record_id="ra", video=video_a, annotator_id="alex", ontology_version=ONTOLOGY.version
    )
    record_b = DatasetRecord(
        record_id="rb", video=video_b, annotator_id="alex", ontology_version=ONTOLOGY.version
    )
    manifest = _manifest(["ra", "rb"], splits={"va": SplitName.TRAIN, "vb": SplitName.TEST})
    report = validate_dataset(manifest, [record_a, record_b], tmp_path, ONTOLOGY)
    assert not report.is_valid
    assert any(i.rule == IssueRule.LEAKED_PLAYER_ACROSS_SPLITS for i in report.errors)


def test_warnings_do_not_affect_validity(tmp_path: Path) -> None:
    video = _video()
    path = tmp_path / video.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"actual bytes")  # deliberately mismatched vs. manifest checksum "a"*64
    record = DatasetRecord(
        record_id="r1", video=video, annotator_id="alex", ontology_version=ONTOLOGY.version
    )
    report = validate_dataset(_manifest(["r1"]), [record], tmp_path, ONTOLOGY)
    assert any(
        i.rule == IssueRule.CHECKSUM_MISMATCH and i.severity == IssueSeverity.WARNING
        for i in report.issues
    )
    assert report.is_valid
