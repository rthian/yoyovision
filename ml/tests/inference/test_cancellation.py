"""Tests for `yoyovision_ml.inference.cancellation`."""

from __future__ import annotations

import time

import pytest
from yoyovision_ml.inference.cancellation import CancellationToken
from yoyovision_ml.inference.errors import PipelineCancelledError, PipelineTimeoutError


def test_default_token_never_cancels() -> None:
    token = CancellationToken()

    token.check("any_stage")  # must not raise


def test_cancel_check_true_raises_cancelled() -> None:
    token = CancellationToken(cancel_check=lambda: True)

    with pytest.raises(PipelineCancelledError, match="cancelled"):
        token.check("pose_extraction")


def test_cancel_check_false_does_not_raise() -> None:
    token = CancellationToken(cancel_check=lambda: False)

    token.check("pose_extraction")  # must not raise


def test_timeout_raises_after_deadline() -> None:
    token = CancellationToken(timeout_s=0.01)
    time.sleep(0.02)

    with pytest.raises(PipelineTimeoutError, match="time budget"):
        token.check("scoring")


def test_timeout_does_not_raise_before_deadline() -> None:
    token = CancellationToken(timeout_s=10.0)

    token.check("scoring")  # must not raise


def test_remaining_s_is_none_without_timeout() -> None:
    token = CancellationToken()

    assert token.remaining_s is None


def test_remaining_s_counts_down() -> None:
    token = CancellationToken(timeout_s=10.0)

    assert token.remaining_s is not None
    assert 0 < token.remaining_s <= 10.0
