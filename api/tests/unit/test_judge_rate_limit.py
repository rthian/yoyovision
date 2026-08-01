"""Unit tests for judge-access rate limiting."""

from __future__ import annotations

import pytest

from yoyovision_api.services.judge_rate_limit import (
    JudgeRateLimitExceeded,
    check_judge_rate_limit,
    reset_judge_rate_limits,
)


@pytest.fixture(autouse=True)
def _clear_buckets() -> None:
    reset_judge_rate_limits()


def test_allows_requests_under_limit() -> None:
    for _ in range(3):
        check_judge_rate_limit("client:abcdefgh", limit_per_minute=3)


def test_blocks_when_limit_exceeded() -> None:
    check_judge_rate_limit("client:abcdefgh", limit_per_minute=2)
    check_judge_rate_limit("client:abcdefgh", limit_per_minute=2)
    with pytest.raises(JudgeRateLimitExceeded):
        check_judge_rate_limit("client:abcdefgh", limit_per_minute=2)


def test_keys_are_isolated() -> None:
    check_judge_rate_limit("client:aaaaaaaa", limit_per_minute=1)
    check_judge_rate_limit("client:bbbbbbbb", limit_per_minute=1)
