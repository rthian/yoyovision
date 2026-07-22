from __future__ import annotations

from datetime import UTC, datetime

from yoyovision_ml.dataset.schema import (
    AnnotationProvenance,
    DatasetRecord,
    DatasetVideo,
    SplitName,
    TrickEventAnnotation,
)
from yoyovision_ml.dataset.stats import compute_annotator_agreement, compute_dataset_statistics
from yoyovision_ml.domain import EventFamily, Outcome, Source

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _provenance(annotator_id: str) -> AnnotationProvenance:
    return AnnotationProvenance(annotator_id=annotator_id, source=Source.HUMAN, annotated_at=NOW)


def _video(video_id: str, player_id: str, duration_ms: int = 10_000) -> DatasetVideo:
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
    event_id: str,
    family: EventFamily,
    start_ms: int,
    end_ms: int,
    annotator_id: str,
    outcome: Outcome = Outcome.SUCCESS,
) -> TrickEventAnnotation:
    return TrickEventAnnotation(
        event_id=event_id,
        label="basic_mount",
        family=family,
        start_ms=start_ms,
        end_ms=end_ms,
        outcome=outcome,
        provenance=_provenance(annotator_id),
    )


def test_compute_dataset_statistics_counts_are_correct() -> None:
    video = _video("v1", "p1")
    record = DatasetRecord(
        record_id="r1",
        video=video,
        annotator_id="alex",
        ontology_version="dataset-ontology-v1",
        trick_events=[
            _event("e1", EventFamily.MOUNT, 0, 500, "alex"),
            _event("e2", EventFamily.HOP, 600, 1200, "alex", outcome=Outcome.MISS),
        ],
    )
    statistics = compute_dataset_statistics([record])
    assert statistics.video_count == 1
    assert statistics.player_count == 1
    assert statistics.event_count == 2
    assert statistics.events_by_family[EventFamily.MOUNT] == 1
    assert statistics.events_by_outcome[Outcome.MISS] == 1
    assert statistics.total_duration_ms == 10_000


def test_compute_dataset_statistics_counts_video_duration_once_per_video() -> None:
    video = _video("v1", "p1")
    record_a = DatasetRecord(
        record_id="ra", video=video, annotator_id="alex", ontology_version="dataset-ontology-v1"
    )
    record_b = DatasetRecord(
        record_id="rb", video=video, annotator_id="bo", ontology_version="dataset-ontology-v1"
    )
    statistics = compute_dataset_statistics([record_a, record_b])
    assert statistics.video_count == 1
    assert statistics.total_duration_ms == 10_000
    assert statistics.record_count == 2


def test_compute_dataset_statistics_with_splits() -> None:
    video_a = _video("va", "p1")
    video_b = _video("vb", "p2")
    record_a = DatasetRecord(
        record_id="ra", video=video_a, annotator_id="alex", ontology_version="dataset-ontology-v1"
    )
    record_b = DatasetRecord(
        record_id="rb", video=video_b, annotator_id="bo", ontology_version="dataset-ontology-v1"
    )
    splits = {"va": SplitName.TRAIN, "vb": SplitName.VAL}
    statistics = compute_dataset_statistics([record_a, record_b], splits)
    assert statistics.videos_by_split[SplitName.TRAIN] == 1
    assert statistics.videos_by_split[SplitName.VAL] == 1


def test_compute_annotator_agreement_matches_overlapping_same_family_events() -> None:
    video = _video("v1", "p1")
    record_a = DatasetRecord(
        record_id="ra",
        video=video,
        annotator_id="alex",
        ontology_version="dataset-ontology-v1",
        trick_events=[_event("ea1", EventFamily.MOUNT, 500, 1500, "alex")],
    )
    record_b = DatasetRecord(
        record_id="rb",
        video=video,
        annotator_id="bo",
        ontology_version="dataset-ontology-v1",
        trick_events=[_event("eb1", EventFamily.MOUNT, 550, 1550, "bo")],
    )
    agreements = compute_annotator_agreement([record_a, record_b])
    assert len(agreements) == 1
    assert agreements[0].matched_event_count == 1
    assert agreements[0].unmatched_event_count == 0
    assert agreements[0].agreement_ratio == 1.0
    assert agreements[0].outcome_agreement_ratio == 1.0


def test_compute_annotator_agreement_flags_unmatched_events() -> None:
    video = _video("v1", "p1")
    record_a = DatasetRecord(
        record_id="ra",
        video=video,
        annotator_id="alex",
        ontology_version="dataset-ontology-v1",
        trick_events=[_event("ea1", EventFamily.MOUNT, 500, 1500, "alex")],
    )
    record_b = DatasetRecord(
        record_id="rb",
        video=video,
        annotator_id="bo",
        ontology_version="dataset-ontology-v1",
        trick_events=[_event("eb1", EventFamily.MOUNT, 8000, 9000, "bo")],
    )
    agreements = compute_annotator_agreement([record_a, record_b])
    assert agreements[0].matched_event_count == 0
    assert agreements[0].unmatched_event_count == 2
    assert agreements[0].agreement_ratio == 0.0


def test_compute_annotator_agreement_excludes_adjudicated_records() -> None:
    video = _video("v1", "p1")
    adjudicated = DatasetRecord(
        record_id="radj",
        video=video,
        annotator_id="alex",
        is_adjudicated=True,
        ontology_version="dataset-ontology-v1",
        trick_events=[_event("eadj1", EventFamily.MOUNT, 500, 1500, "alex")],
    )
    assert compute_annotator_agreement([adjudicated]) == []
