from __future__ import annotations

import pytest
from pydantic import ValidationError

from yoyovision_ml.dataset.ontology import EventOntology, default_ontology
from yoyovision_ml.domain import EventFamily


def test_default_ontology_loads() -> None:
    ontology = default_ontology()
    assert ontology.version == "dataset-ontology-v1"
    assert len(ontology.labels) > 0


def test_every_domain_event_family_has_at_least_one_label() -> None:
    ontology = default_ontology()
    covered_families = {entry.family for entry in ontology.labels}
    assert covered_families == set(EventFamily)


def test_is_known_label_true_for_canonical_and_alias() -> None:
    ontology = default_ontology()
    assert ontology.is_known_label("basic_mount")
    assert ontology.is_known_label("basic_start")  # alias of basic_mount


def test_is_known_label_false_for_unknown() -> None:
    ontology = default_ontology()
    assert not ontology.is_known_label("totally_made_up_trick")


def test_family_for_label() -> None:
    ontology = default_ontology()
    assert ontology.family_for_label("eli_hop") == EventFamily.HOP
    assert ontology.family_for_label("totally_made_up_trick") is None


def test_allows_overlap_respects_configured_families() -> None:
    ontology = default_ontology()
    assert ontology.allows_overlap(EventFamily.BODY_TRICK)
    assert not ontology.allows_overlap(EventFamily.MOUNT)


def test_duplicate_label_or_alias_rejected() -> None:
    with pytest.raises(ValidationError):
        EventOntology(
            version="broken",
            labels=[
                {"label": "a", "family": "mount", "aliases": ["shared"]},
                {"label": "b", "family": "hop", "aliases": ["shared"]},
            ],
            visibility_states=["visible"],
        )
