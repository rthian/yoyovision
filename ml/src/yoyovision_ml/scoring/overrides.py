"""Prompt D requirement 6: "Support per-event manual overrides."

Applies a list of `EventOverride`s to a list of persisted `AnalysisEvent`s,
producing both the corrected events and a human-readable audit trail. Every
override is either applied and logged, or explicitly rejected and logged --
nothing is ever silently dropped or silently folded into the score.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from yoyovision_ml.domain import AnalysisEvent, DifficultyBand, EventFamily, Outcome, ReviewStatus
from yoyovision_ml.scoring.types import OVERRIDABLE_EVENT_FIELDS, EventOverride

_FIELD_PARSERS = {
    "label": str,
    "family": EventFamily,
    "start_ms": int,
    "end_ms": int,
    "outcome": Outcome,
    "difficulty_band": DifficultyBand,
    "confidence": float,
}


def _parse_field_value(field_name: str, raw_value: str) -> object:
    parser = _FIELD_PARSERS[field_name]
    try:
        return parser(raw_value)
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"Cannot parse override value '{raw_value}' for field '{field_name}': {exc}"
        ) from exc


def apply_overrides(
    events: Sequence[AnalysisEvent], overrides: Sequence[EventOverride]
) -> tuple[list[AnalysisEvent], list[str]]:
    """Applies `overrides` to (deep copies of) `events`, in list order.

    Returns the corrected event list (original order, un-overridden events
    untouched) and an audit-log list of human-readable strings describing
    every override that was applied, skipped (unknown `event_id`), or
    rejected (disallowed `field_name` / unparseable value) -- the pipeline
    surfaces this log verbatim in `ScoringPipelineResult.override_audit_log`.
    """
    events_by_id = {event.id: deepcopy(event) for event in events}
    order = [event.id for event in events]
    audit_log: list[str] = []

    for override in overrides:
        if override.field_name not in OVERRIDABLE_EVENT_FIELDS:
            audit_log.append(
                f"REJECTED override for event_id={override.event_id}: field "
                f"'{override.field_name}' is not overridable "
                f"(allowed: {sorted(OVERRIDABLE_EVENT_FIELDS)})."
            )
            continue

        event = events_by_id.get(override.event_id)
        if event is None:
            audit_log.append(
                f"SKIPPED override for unknown event_id={override.event_id} "
                f"(field '{override.field_name}')."
            )
            continue

        try:
            parsed_value = _parse_field_value(override.field_name, override.overridden_value)
        except ValueError as exc:
            audit_log.append(f"REJECTED override for event_id={override.event_id}: {exc}")
            continue

        current_value = str(getattr(event, override.field_name))
        setattr(event, override.field_name, parsed_value)
        if event.review_status != ReviewStatus.REJECTED:
            event.review_status = ReviewStatus.EDITED

        audit_log.append(
            f"event_id={override.event_id}: {override.field_name} "
            f"'{current_value}' -> '{override.overridden_value}' "
            f"by {override.overridden_by} at {override.overridden_at.isoformat()}"
            f"{f' ({override.reason})' if override.reason else ''}."
        )

    return [events_by_id[event_id] for event_id in order], audit_log
