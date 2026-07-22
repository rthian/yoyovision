from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from yoyovision_ml.dataset.schema import (
    AnnotationProvenance,
    DatasetRecord,
    DatasetVideo,
    NormalizedBBox,
    StringMaskFrame,
    TrickEventAnnotation,
    VisibilityState,
    YoyoFrameAnnotation,
)
from yoyovision_ml.domain import EventFamily, Outcome, Source

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _provenance() -> AnnotationProvenance:
    return AnnotationProvenance(annotator_id="alex", source=Source.HUMAN, annotated_at=NOW)


def _video(**overrides: object) -> DatasetVideo:
    defaults: dict[str, object] = dict(
        video_id="v1",
        player_id="p1",
        relative_path="videos/v1.mp4",
        checksum_sha256="a" * 64,
        duration_ms=10_000,
        width=1920,
        height=1080,
        source_fps=30.0,
    )
    defaults.update(overrides)
    return DatasetVideo.model_validate(defaults)


def test_valid_video_round_trips_through_json() -> None:
    video = _video()
    restored = DatasetVideo.model_validate_json(video.model_dump_json())
    assert restored == video


def test_video_rejects_path_traversal() -> None:
    with pytest.raises(ValidationError):
        _video(relative_path="../../etc/passwd")


def test_video_rejects_absolute_path() -> None:
    with pytest.raises(ValidationError):
        _video(relative_path="/etc/passwd")


def test_variable_fps_requires_frame_timestamps() -> None:
    with pytest.raises(ValidationError):
        _video(is_variable_fps=True)


def test_variable_fps_with_frame_timestamps_is_valid() -> None:
    video = _video(is_variable_fps=True, frame_timestamps_ms=[0, 33, 67, 100])
    assert video.frame_timestamps_ms == [0, 33, 67, 100]


def test_frame_timestamps_must_be_non_decreasing() -> None:
    with pytest.raises(ValidationError):
        _video(is_variable_fps=True, frame_timestamps_ms=[0, 100, 33])


def test_trick_event_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError):
        TrickEventAnnotation(
            event_id="e1",
            label="basic_mount",
            family=EventFamily.MOUNT,
            start_ms=1000,
            end_ms=500,
            outcome=Outcome.SUCCESS,
            provenance=_provenance(),
        )


def test_trick_event_rejects_zero_length_interval() -> None:
    with pytest.raises(ValidationError):
        TrickEventAnnotation(
            event_id="e1",
            label="basic_mount",
            family=EventFamily.MOUNT,
            start_ms=1000,
            end_ms=1000,
            outcome=Outcome.SUCCESS,
            provenance=_provenance(),
        )


def test_yoyo_frame_requires_position_when_visible() -> None:
    with pytest.raises(ValidationError):
        YoyoFrameAnnotation(frame_ms=0, visibility=VisibilityState.VISIBLE)


def test_yoyo_frame_allows_missing_position_when_fully_occluded() -> None:
    frame = YoyoFrameAnnotation(frame_ms=0, visibility=VisibilityState.FULLY_OCCLUDED)
    assert frame.point is None
    assert frame.bbox is None


def test_bbox_rejects_out_of_frame_extent() -> None:
    with pytest.raises(ValidationError):
        NormalizedBBox(x=0.9, y=0.9, width=0.5, height=0.5)


def test_string_mask_requires_mask_key_when_observable() -> None:
    with pytest.raises(ValidationError):
        StringMaskFrame(frame_ms=0, observable=True, mask_key=None)


def test_string_mask_forbids_mask_key_when_unobservable() -> None:
    with pytest.raises(ValidationError):
        StringMaskFrame(frame_ms=0, observable=False, mask_key="mask_0.png")


def test_string_mask_valid_when_unobservable_and_no_mask_key() -> None:
    frame = StringMaskFrame(frame_ms=0, observable=False, mask_key=None)
    assert frame.mask_key is None


def test_adjudication_requires_adjudicated_by() -> None:
    with pytest.raises(ValidationError):
        AnnotationProvenance(
            annotator_id="alex",
            source=Source.HUMAN,
            annotated_at=NOW,
            is_adjudicated=True,
        )


def test_record_rejects_duplicate_event_ids() -> None:
    video = _video()
    event_kwargs = dict(
        label="basic_mount",
        family=EventFamily.MOUNT,
        start_ms=0,
        end_ms=500,
        outcome=Outcome.SUCCESS,
        provenance=_provenance(),
    )
    with pytest.raises(ValidationError):
        DatasetRecord(
            record_id="r1",
            video=video,
            annotator_id="alex",
            ontology_version="dataset-ontology-v1",
            trick_events=[
                TrickEventAnnotation(event_id="dup", **event_kwargs),
                TrickEventAnnotation(event_id="dup", **event_kwargs),
            ],
        )


def test_record_round_trips_through_json() -> None:
    video = _video()
    record = DatasetRecord(
        record_id="r1",
        video=video,
        annotator_id="alex",
        ontology_version="dataset-ontology-v1",
        trick_events=[
            TrickEventAnnotation(
                event_id="e1",
                label="basic_mount",
                family=EventFamily.MOUNT,
                start_ms=0,
                end_ms=500,
                outcome=Outcome.SUCCESS,
                provenance=_provenance(),
            )
        ],
    )
    restored = DatasetRecord.model_validate_json(record.model_dump_json())
    assert restored == record
