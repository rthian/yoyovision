#!/usr/bin/env python
"""Generates the synthetic sample dataset under `ml/sample_data/dataset_v1/`.

Re-run this script (`python scripts/generate_sample_dataset.py` from `ml/`)
after any `dataset/schema.py` change to keep the sample dataset a valid,
up-to-date exercise of every schema field. The dataset is entirely
synthetic: placeholder video "files" (short marker text, not real video
bytes) and hand-authored annotation values -- see
`sample_data/dataset_v1/README.md`.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from yoyovision_ml.dataset.io import save_manifest, save_record
from yoyovision_ml.dataset.ontology import default_ontology
from yoyovision_ml.dataset.schema import (
    AnnotationProvenance,
    DatasetManifest,
    DatasetRecord,
    DatasetVideo,
    DeductionAnnotation,
    FreestyleEvaluationAnnotation,
    JudgeClickAnnotation,
    NormalizedBBox,
    NormalizedPoint,
    TrickEventAnnotation,
    VisibilityState,
    YoyoFrameAnnotation,
)
from yoyovision_ml.domain import DeductionType, DifficultyBand, Outcome, Source

DATASET_DIR = Path(__file__).parent.parent / "sample_data" / "dataset_v1"
ONTOLOGY_VERSION = default_ontology().version
NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


def _placeholder_video(relative_name: str) -> tuple[str, str]:
    """Writes a short, clearly-labelled placeholder file (NOT real video
    bytes) and returns (relative_path, sha256_of_actual_bytes)."""
    path = DATASET_DIR / "videos" / relative_name
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"YOYOVISION SAMPLE DATASET PLACEHOLDER -- NOT REAL VIDEO CONTENT\nfile={relative_name}\n"
    ).encode()
    path.write_bytes(content)
    checksum = hashlib.sha256(content).hexdigest()
    return f"videos/{relative_name}", checksum


def _provenance(annotator_id: str, *, adjudicated_by: str | None = None) -> AnnotationProvenance:
    return AnnotationProvenance(
        annotator_id=annotator_id,
        source=Source.HUMAN,
        annotated_at=NOW,
        tool="manual",
        is_adjudicated=adjudicated_by is not None,
        adjudicated_by=adjudicated_by,
        adjudication_notes=(
            "Merged from annotator_alex and annotator_bo; kept annotator_alex's boundaries "
            "where both agreed, added the missed catch_miss annotator_bo alone caught."
            if adjudicated_by
            else ""
        ),
    )


def _yoyo_track(n_frames: int, frame_ms_step: int) -> list[YoyoFrameAnnotation]:
    track = []
    for i in range(n_frames):
        frame_ms = i * frame_ms_step
        if i == n_frames // 2:
            # one frame demonstrating a partially-occluded hand pass
            track.append(
                YoyoFrameAnnotation(
                    frame_ms=frame_ms,
                    bbox=NormalizedBBox(x=0.44, y=0.5, width=0.05, height=0.05),
                    visibility=VisibilityState.PARTIALLY_OCCLUDED,
                    confidence=None,
                )
            )
        else:
            x = 0.4 + 0.1 * (i / max(n_frames - 1, 1))
            track.append(
                YoyoFrameAnnotation(
                    frame_ms=frame_ms,
                    point=NormalizedPoint(x=round(x, 4), y=0.5),
                    visibility=VisibilityState.VISIBLE,
                    confidence=None,
                )
            )
    return track


def _video_1_records() -> tuple[DatasetVideo, list[DatasetRecord]]:
    relative_path, checksum = _placeholder_video("sample_video_001.mp4")
    video = DatasetVideo(
        video_id="sample_video_001",
        player_id="player_a",
        relative_path=relative_path,
        checksum_sha256=checksum,
        duration_ms=20_000,
        width=1920,
        height=1080,
        source_fps=30.0,
        notes=(
            "Synthetic sample video 1 for player_a (has two raw annotator passes + adjudication)."
        ),
    )

    shared_track = _yoyo_track(6, 3000)

    raw_alex = DatasetRecord(
        record_id="sample_video_001__annotator_alex",
        video=video,
        annotator_id="annotator_alex",
        is_adjudicated=False,
        ontology_version=ONTOLOGY_VERSION,
        yoyo_track=shared_track,
        trick_events=[
            TrickEventAnnotation(
                event_id="sv001_alex_evt1",
                label="basic_mount",
                family="mount",
                start_ms=500,
                end_ms=1500,
                outcome=Outcome.SUCCESS,
                difficulty_band=DifficultyBand.BASIC,
                provenance=_provenance("annotator_alex"),
            ),
            TrickEventAnnotation(
                event_id="sv001_alex_evt2",
                label="eli_hop",
                family="hop",
                start_ms=2000,
                end_ms=3200,
                outcome=Outcome.SUCCESS,
                difficulty_band=DifficultyBand.INTERMEDIATE,
                provenance=_provenance("annotator_alex"),
            ),
        ],
    )

    raw_bo = DatasetRecord(
        record_id="sample_video_001__annotator_bo",
        video=video,
        annotator_id="annotator_bo",
        is_adjudicated=False,
        ontology_version=ONTOLOGY_VERSION,
        yoyo_track=shared_track,
        trick_events=[
            TrickEventAnnotation(
                event_id="sv001_bo_evt1",
                label="basic_mount",
                family="mount",
                start_ms=550,
                end_ms=1550,
                outcome=Outcome.SUCCESS,
                difficulty_band=DifficultyBand.BASIC,
                provenance=_provenance("annotator_bo"),
            ),
            TrickEventAnnotation(
                event_id="sv001_bo_evt2",
                label="eli_hop",
                family="hop",
                start_ms=2050,
                end_ms=3250,
                outcome=Outcome.SUCCESS,
                difficulty_band=DifficultyBand.INTERMEDIATE,
                provenance=_provenance("annotator_bo"),
            ),
            TrickEventAnnotation(
                event_id="sv001_bo_evt3",
                label="missed_catch",
                family="catch_miss",
                start_ms=15_000,
                end_ms=15_400,
                outcome=Outcome.MISS,
                provenance=_provenance("annotator_bo"),
                notes="Only annotator_bo caught this missed catch near the end of the routine.",
            ),
        ],
    )

    adjudicated = DatasetRecord(
        record_id="sample_video_001__adjudicated",
        video=video,
        annotator_id="annotator_alex",
        is_adjudicated=True,
        ontology_version=ONTOLOGY_VERSION,
        yoyo_track=shared_track,
        trick_events=[
            TrickEventAnnotation(
                event_id="sv001_adj_evt1",
                label="basic_mount",
                family="mount",
                start_ms=500,
                end_ms=1500,
                outcome=Outcome.SUCCESS,
                difficulty_band=DifficultyBand.BASIC,
                provenance=_provenance("annotator_alex", adjudicated_by="lead_reviewer"),
            ),
            TrickEventAnnotation(
                event_id="sv001_adj_evt2",
                label="eli_hop",
                family="hop",
                start_ms=2000,
                end_ms=3200,
                outcome=Outcome.SUCCESS,
                difficulty_band=DifficultyBand.INTERMEDIATE,
                provenance=_provenance("annotator_alex", adjudicated_by="lead_reviewer"),
            ),
            TrickEventAnnotation(
                event_id="sv001_adj_evt3",
                label="double_or_nothing",
                family="whip_catch",
                start_ms=4000,
                end_ms=5000,
                outcome=Outcome.SUCCESS,
                difficulty_band=DifficultyBand.ADVANCED,
                confidence=0.82,
                provenance=_provenance("annotator_alex", adjudicated_by="lead_reviewer"),
            ),
            TrickEventAnnotation(
                event_id="sv001_adj_evt4",
                label="missed_catch",
                family="catch_miss",
                start_ms=15_000,
                end_ms=15_400,
                outcome=Outcome.MISS,
                provenance=_provenance("annotator_alex", adjudicated_by="lead_reviewer"),
                notes="Adopted from annotator_bo's pass; annotator_alex initially missed this.",
            ),
        ],
        deductions=[
            DeductionAnnotation(
                deduction_id="sv001_adj_ded1",
                type=DeductionType.OTHER,
                timestamp_ms=15_400,
                quantity=1,
                provenance=_provenance("annotator_alex", adjudicated_by="lead_reviewer"),
                notes="Minor form deduction associated with the missed catch.",
            ),
        ],
        judge_clicks=[
            JudgeClickAnnotation(
                click_id="sv001_click1",
                judge_id="judge_1",
                timestamp_ms=4050,
                associated_label="double_or_nothing",
                notes="Judge's real-time perceived start of the whip.",
            ),
        ],
        freestyle_evaluations=[
            FreestyleEvaluationAnnotation(
                judge_id="judge_1",
                execution=7.5,
                control=7.0,
                trick_diversity=8.0,
                space_use_emphasis=6.5,
                music_choreography=7.0,
                music_construction=6.0,
                body_control=7.5,
                showmanship=8.0,
                provenance=_provenance("judge_1"),
            ),
        ],
    )

    return video, [raw_alex, raw_bo, adjudicated]


def _video_2_record() -> tuple[DatasetVideo, DatasetRecord]:
    relative_path, checksum = _placeholder_video("sample_video_002.mp4")
    video = DatasetVideo(
        video_id="sample_video_002",
        player_id="player_a",
        relative_path=relative_path,
        checksum_sha256=checksum,
        duration_ms=15_000,
        width=1920,
        height=1080,
        source_fps=30.0,
        notes=(
            "Synthetic sample video 2 for player_a (single-annotator pass, no adjudication needed)."
        ),
    )
    record = DatasetRecord(
        record_id="sample_video_002__annotator_alex",
        video=video,
        annotator_id="annotator_alex",
        is_adjudicated=False,
        ontology_version=ONTOLOGY_VERSION,
        yoyo_track=_yoyo_track(4, 3000),
        trick_events=[
            TrickEventAnnotation(
                event_id="sv002_evt1",
                label="basic_mount",
                family="mount",
                start_ms=300,
                end_ms=1200,
                outcome=Outcome.SUCCESS,
                difficulty_band=DifficultyBand.BASIC,
                provenance=_provenance("annotator_alex"),
            ),
            TrickEventAnnotation(
                event_id="sv002_evt2",
                label="unclassified_element",
                family="unknown_technical_element",
                start_ms=6000,
                end_ms=6800,
                outcome=Outcome.UNCERTAIN,
                provenance=_provenance("annotator_alex"),
                notes="Annotator was unsure whether this was a novel horizontal variant.",
            ),
            TrickEventAnnotation(
                event_id="sv002_evt3",
                label="yoyo_stopped_spinning",
                family="yoyo_stop",
                start_ms=10_000,
                end_ms=10_200,
                outcome=Outcome.MISS,
                provenance=_provenance("annotator_alex"),
            ),
        ],
        deductions=[
            DeductionAnnotation(
                deduction_id="sv002_ded1",
                type=DeductionType.YOYO_STOP,
                timestamp_ms=10_200,
                quantity=1,
                provenance=_provenance("annotator_alex"),
            ),
        ],
    )
    return video, record


def _video_3_record() -> tuple[DatasetVideo, DatasetRecord]:
    relative_path, checksum = _placeholder_video("sample_video_003.mp4")
    video = DatasetVideo(
        video_id="sample_video_003",
        player_id="player_b",
        relative_path=relative_path,
        checksum_sha256=checksum,
        duration_ms=18_000,
        width=1280,
        height=720,
        source_fps=25.0,
        notes="Synthetic sample video for player_b, on a different camera (720p/25fps).",
    )
    record = DatasetRecord(
        record_id="sample_video_003__adjudicated",
        video=video,
        annotator_id="annotator_bo",
        is_adjudicated=True,
        ontology_version=ONTOLOGY_VERSION,
        yoyo_track=_yoyo_track(5, 3000),
        trick_events=[
            TrickEventAnnotation(
                event_id="sv003_evt1",
                label="brain_twister_mount",
                family="mount",
                start_ms=400,
                end_ms=1600,
                outcome=Outcome.SUCCESS,
                difficulty_band=DifficultyBand.INTERMEDIATE,
                provenance=_provenance("annotator_bo", adjudicated_by="lead_reviewer"),
            ),
            TrickEventAnnotation(
                event_id="sv003_evt2",
                label="trapeze_bind",
                family="bind",
                start_ms=16_500,
                end_ms=17_800,
                outcome=Outcome.SUCCESS,
                difficulty_band=DifficultyBand.BASIC,
                provenance=_provenance("annotator_bo", adjudicated_by="lead_reviewer"),
            ),
        ],
        freestyle_evaluations=[
            FreestyleEvaluationAnnotation(
                judge_id="judge_2",
                execution=6.0,
                control=6.5,
                trick_diversity=5.5,
                space_use_emphasis=None,
                music_choreography=None,
                music_construction=None,
                body_control=6.0,
                showmanship=None,
                provenance=_provenance("judge_2"),
                notes=(
                    "Partial scorecard: judge_2 did not evaluate space use / music / showmanship."
                ),
            ),
        ],
    )
    return video, record


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    video_1, records_1 = _video_1_records()
    video_2, record_2 = _video_2_record()
    video_3, record_3 = _video_3_record()

    all_records = [*records_1, record_2, record_3]
    record_paths = [save_record(DATASET_DIR, r) for r in all_records]

    manifest = DatasetManifest(
        dataset_version="sample-v1",
        ontology_version=ONTOLOGY_VERSION,
        created_at=NOW,
        video_ids=[video_1.video_id, video_2.video_id, video_3.video_id],
        record_paths=[str(p.relative_to(DATASET_DIR)) for p in record_paths],
        notes=(
            "Synthetic sample dataset for YoYoVision Prompt A. All video files under "
            "videos/ are placeholder text, not real footage. See README.md in this "
            "directory."
        ),
    )
    save_manifest(DATASET_DIR, manifest)

    readme_path = DATASET_DIR / "README.md"
    readme_path.write_text(
        "# Sample dataset (synthetic)\n\n"
        "Generated by `ml/scripts/generate_sample_dataset.py`. Every file under "
        "`videos/` is a short placeholder text marker, **not real video content** -- "
        "there is no real annotated 1A footage in this repository yet. Every "
        "annotation value (events, deductions, judge clicks, Freestyle Evaluation "
        "scores) is hand-authored to exercise the dataset schema end to end, not "
        "measured from real performance.\n\n"
        "Regenerate with:\n\n"
        "```bash\n"
        "cd ml && python scripts/generate_sample_dataset.py\n"
        "```\n\n"
        "Validate with:\n\n"
        "```bash\n"
        "cd ml && python -m yoyovision_ml.dataset.cli validate sample_data/dataset_v1\n"
        "```\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(all_records)} records and manifest.json to {DATASET_DIR}")


if __name__ == "__main__":
    main()
