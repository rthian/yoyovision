"""Prompt D requirement 3: "Support multiple scoring profiles: practice,
judge_assist, research." Tests `scoring.profiles`."""

from __future__ import annotations

import pytest

from yoyovision_ml.domain import ReviewStatus
from yoyovision_ml.scoring.profiles import (
    PROFILE_CONFIGS,
    ScoringProfile,
    get_profile_config,
    minimum_review_status_ok,
)


def test_all_three_required_profiles_exist() -> None:
    assert {p.value for p in ScoringProfile} == {"practice", "judge_assist", "research"}


@pytest.mark.parametrize("profile", list(ScoringProfile))
def test_every_profile_has_a_config(profile: ScoringProfile) -> None:
    config = get_profile_config(profile)
    assert config.profile == profile


def test_practice_profile_never_requires_confirmation_and_gives_full_credit() -> None:
    config = PROFILE_CONFIGS[ScoringProfile.PRACTICE]
    assert config.require_confirmed_events is False
    assert config.repeated_element_policy_override == "full_credit"
    assert config.compute_confidence_interval is False


def test_judge_assist_profile_matches_pre_prompt_d_defaults() -> None:
    config = PROFILE_CONFIGS[ScoringProfile.JUDGE_ASSIST]
    assert config.require_confirmed_events is False
    assert config.repeated_element_policy_override is None
    assert config.require_multiple_judges is False
    assert config.compute_confidence_interval is True


def test_research_profile_is_strictest() -> None:
    config = PROFILE_CONFIGS[ScoringProfile.RESEARCH]
    assert config.require_confirmed_events is True
    assert config.require_multiple_judges is True
    assert config.repeated_element_policy_override == "decay_all_families"
    assert config.compute_confidence_interval is True


def test_no_profile_disables_automatic_fe_estimators() -> None:
    """`use_automatic_fe_estimators` gates estimator *use*, not showmanship
    manual-only-ness -- every packaged profile still allows estimators for
    the six eligible categories."""
    for config in PROFILE_CONFIGS.values():
        assert config.use_automatic_fe_estimators is True


@pytest.mark.parametrize(
    ("review_status", "require_confirmed", "expected"),
    [
        (ReviewStatus.REJECTED, False, False),
        (ReviewStatus.REJECTED, True, False),
        (ReviewStatus.PENDING, False, True),
        (ReviewStatus.PENDING, True, False),
        (ReviewStatus.EDITED, True, False),
        (ReviewStatus.CONFIRMED, True, True),
        (ReviewStatus.CONFIRMED, False, True),
    ],
)
def test_minimum_review_status_ok_matrix(
    review_status: ReviewStatus, require_confirmed: bool, expected: bool
) -> None:
    config = get_profile_config(ScoringProfile.JUDGE_ASSIST).model_copy(
        update={"require_confirmed_events": require_confirmed}
    )
    assert minimum_review_status_ok(review_status, config) is expected


def test_research_profile_rejects_everything_but_confirmed() -> None:
    config = get_profile_config(ScoringProfile.RESEARCH)
    assert minimum_review_status_ok(ReviewStatus.PENDING, config) is False
    assert minimum_review_status_ok(ReviewStatus.EDITED, config) is False
    assert minimum_review_status_ok(ReviewStatus.CONFIRMED, config) is True
