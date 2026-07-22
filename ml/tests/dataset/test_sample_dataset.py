"""Integration test: the committed synthetic sample dataset must stay valid.

Also serves as the "validation report" deliverable's regression guard --
if a future schema/ontology change breaks the sample dataset, this test
fails loudly instead of the sample dataset silently going stale.
"""

from __future__ import annotations

from pathlib import Path

from yoyovision_ml.dataset.io import load_dataset
from yoyovision_ml.dataset.ontology import default_ontology
from yoyovision_ml.dataset.splits import generate_player_grouped_splits
from yoyovision_ml.dataset.stats import compute_annotator_agreement, compute_dataset_statistics
from yoyovision_ml.dataset.validators import validate_dataset

SAMPLE_DATASET_DIR = Path(__file__).parent.parent.parent / "sample_data" / "dataset_v1"


def test_sample_dataset_exists() -> None:
    assert SAMPLE_DATASET_DIR.exists(), (
        f"{SAMPLE_DATASET_DIR} is missing -- run "
        "`python scripts/generate_sample_dataset.py` from the ml/ directory."
    )


def test_sample_dataset_loads_and_is_valid() -> None:
    manifest, records = load_dataset(SAMPLE_DATASET_DIR)
    ontology = default_ontology()
    report = validate_dataset(manifest, records, SAMPLE_DATASET_DIR, ontology)
    assert report.is_valid, [str(issue) for issue in report.errors]


def test_sample_dataset_has_multi_annotator_and_adjudicated_records() -> None:
    _, records = load_dataset(SAMPLE_DATASET_DIR)
    adjudicated = [r for r in records if r.is_adjudicated]
    raw_passes = [r for r in records if not r.is_adjudicated]
    assert len(adjudicated) >= 1
    assert len(raw_passes) >= 2


def test_sample_dataset_agreement_stats_run_without_error() -> None:
    _, records = load_dataset(SAMPLE_DATASET_DIR)
    agreements = compute_annotator_agreement(records)
    assert len(agreements) >= 1


def test_sample_dataset_stats_run_without_error() -> None:
    _, records = load_dataset(SAMPLE_DATASET_DIR)
    statistics = compute_dataset_statistics(records)
    assert statistics.video_count >= 1
    assert statistics.event_count >= 1


def test_sample_dataset_splits_without_leakage() -> None:
    _, records = load_dataset(SAMPLE_DATASET_DIR)
    videos = list({r.video.video_id: r.video for r in records}.values())
    assignment = generate_player_grouped_splits(videos, seed=42)
    for video in videos:
        assert assignment.video_splits[video.video_id] == assignment.player_splits[video.player_id]
