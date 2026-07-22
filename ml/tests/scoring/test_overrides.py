"""Prompt D requirement 6: "Support per-event manual overrides." Tests
`scoring.overrides.apply_overrides`."""

from __future__ import annotations

from conftest import make_analysis_event, make_event_override

from yoyovision_ml.domain import DifficultyBand, EventFamily, Outcome, ReviewStatus
from yoyovision_ml.scoring.overrides import apply_overrides


def test_applying_a_valid_override_updates_field_and_marks_edited() -> None:
    events = [make_analysis_event("evt-1", outcome=Outcome.MISS)]
    overrides = [
        make_event_override(
            event_id="evt-1",
            field_name="outcome",
            original_value="miss",
            overridden_value="success",
        )
    ]

    corrected, audit_log = apply_overrides(events, overrides)

    assert corrected[0].outcome == Outcome.SUCCESS
    assert corrected[0].review_status == ReviewStatus.EDITED
    assert len(audit_log) == 1
    assert "evt-1" in audit_log[0]
    assert "outcome" in audit_log[0]


def test_original_events_list_is_never_mutated() -> None:
    events = [make_analysis_event("evt-1", outcome=Outcome.MISS)]
    overrides = [
        make_event_override(
            event_id="evt-1",
            field_name="outcome",
            original_value="miss",
            overridden_value="success",
        )
    ]

    apply_overrides(events, overrides)

    assert events[0].outcome == Outcome.MISS
    assert events[0].review_status == ReviewStatus.PENDING


def test_output_preserves_original_event_order() -> None:
    events = [
        make_analysis_event("evt-1"),
        make_analysis_event("evt-2"),
        make_analysis_event("evt-3"),
    ]
    corrected, _ = apply_overrides(events, [])
    assert [e.id for e in corrected] == ["evt-1", "evt-2", "evt-3"]


def test_unknown_event_id_is_skipped_and_logged_not_raised() -> None:
    events = [make_analysis_event("evt-1")]
    overrides = [
        make_event_override(
            event_id="does-not-exist", field_name="outcome", overridden_value="success"
        )
    ]

    corrected, audit_log = apply_overrides(events, overrides)

    assert corrected[0].outcome == events[0].outcome
    assert len(audit_log) == 1
    assert "SKIPPED" in audit_log[0]
    assert "does-not-exist" in audit_log[0]


def test_disallowed_field_name_is_rejected_and_logged_not_raised() -> None:
    events = [make_analysis_event("evt-1")]
    overrides = [
        make_event_override(event_id="evt-1", field_name="id", overridden_value="hacked-id")
    ]

    corrected, audit_log = apply_overrides(events, overrides)

    assert corrected[0].id == "evt-1"
    assert len(audit_log) == 1
    assert "REJECTED" in audit_log[0]
    assert "not overridable" in audit_log[0]


def test_unparseable_value_is_rejected_and_logged_not_raised() -> None:
    events = [make_analysis_event("evt-1")]
    overrides = [
        make_event_override(
            event_id="evt-1", field_name="difficulty_band", overridden_value="not-a-real-band"
        )
    ]

    corrected, audit_log = apply_overrides(events, overrides)

    assert corrected[0].difficulty_band == DifficultyBand.BASIC
    assert len(audit_log) == 1
    assert "REJECTED" in audit_log[0]
    assert "Cannot parse" in audit_log[0]


def test_rejected_event_is_not_forced_back_to_edited() -> None:
    """An event a human already REJECTED keeps that status after an
    override is applied to it -- only non-rejected events flip to EDITED."""
    events = [
        make_analysis_event(
            "evt-1", outcome=Outcome.MISS, review_status=ReviewStatus.REJECTED
        )
    ]
    overrides = [
        make_event_override(
            event_id="evt-1", field_name="outcome", overridden_value="success"
        )
    ]

    corrected, _ = apply_overrides(events, overrides)

    assert corrected[0].outcome == Outcome.SUCCESS
    assert corrected[0].review_status == ReviewStatus.REJECTED


def test_family_field_override_parses_to_enum() -> None:
    events = [make_analysis_event("evt-1", family=EventFamily.MOUNT)]
    overrides = [
        make_event_override(event_id="evt-1", field_name="family", overridden_value="suicide")
    ]

    corrected, _ = apply_overrides(events, overrides)

    assert corrected[0].family == EventFamily.SUICIDE


def test_multiple_overrides_for_same_event_apply_in_order() -> None:
    events = [make_analysis_event("evt-1", outcome=Outcome.MISS, band=DifficultyBand.BASIC)]
    overrides = [
        make_event_override(event_id="evt-1", field_name="outcome", overridden_value="success"),
        make_event_override(
            event_id="evt-1", field_name="difficulty_band", overridden_value="advanced"
        ),
    ]

    corrected, audit_log = apply_overrides(events, overrides)

    assert corrected[0].outcome == Outcome.SUCCESS
    assert corrected[0].difficulty_band == DifficultyBand.ADVANCED
    assert len(audit_log) == 2
