"""Versioned dataset annotation ontology.

Maps specific, human-readable trick labels onto the fixed `EventFamily`
vocabulary (`yoyovision_ml.domain`). Loaded from YAML so new labels can be
added without a code change; every `DatasetRecord` pins the exact
`ontology_version` it was annotated against (see docs/data_model.md's
versioning convention for the scoring `Ruleset`, mirrored here).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from yoyovision_ml.dataset.schema import VisibilityState
from yoyovision_ml.domain import EventFamily

_ONTOLOGY_DIR = Path(__file__).parent / "ontology"
_DEFAULT_ONTOLOGY_FILENAME = "v1.yaml"


class OntologyLabel(BaseModel):
    label: str
    family: EventFamily
    description: str = ""
    aliases: list[str] = Field(default_factory=list)


class EventOntology(BaseModel):
    version: str
    labels: list[OntologyLabel]
    visibility_states: list[VisibilityState]
    overlap_allowed_families: list[EventFamily] = Field(default_factory=list)

    @field_validator("labels")
    @classmethod
    def _no_duplicate_labels_or_aliases(cls, labels: list[OntologyLabel]) -> list[OntologyLabel]:
        seen: set[str] = set()
        for entry in labels:
            names = [entry.label, *entry.aliases]
            for name in names:
                if name in seen:
                    raise ValueError(f"Duplicate ontology label/alias: '{name}'")
                seen.add(name)
        return labels

    def label_index(self) -> dict[str, OntologyLabel]:
        """Maps every canonical label AND every alias to its `OntologyLabel`."""
        index: dict[str, OntologyLabel] = {}
        for entry in self.labels:
            index[entry.label] = entry
            for alias in entry.aliases:
                index[alias] = entry
        return index

    def is_known_label(self, label: str) -> bool:
        return label in self.label_index()

    def family_for_label(self, label: str) -> EventFamily | None:
        entry = self.label_index().get(label)
        return entry.family if entry is not None else None

    def allows_overlap(self, family: EventFamily) -> bool:
        return family in self.overlap_allowed_families


def load_ontology(path: Path) -> EventOntology:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return EventOntology.model_validate(raw)


@lru_cache(maxsize=8)
def default_ontology() -> EventOntology:
    """The packaged v1 ontology."""
    return load_ontology(_ONTOLOGY_DIR / _DEFAULT_ONTOLOGY_FILENAME)


def available_ontology_files() -> list[Path]:
    if not _ONTOLOGY_DIR.exists():
        return []
    return sorted(_ONTOLOGY_DIR.glob("*.yaml"))


def get_ontology_by_version(version: str) -> EventOntology | None:
    for path in available_ontology_files():
        ontology = load_ontology(path)
        if ontology.version == version:
            return ontology
    return None
