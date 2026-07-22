"""Tests for `yoyovision_ml.inference.timing`."""

from __future__ import annotations

import time

from yoyovision_ml.inference.timing import StageTimings


def test_measure_records_a_positive_duration() -> None:
    timings = StageTimings()

    with timings.measure("stage_a"):
        time.sleep(0.001)

    assert "stage_a" in timings.durations_ms
    assert timings.durations_ms["stage_a"] > 0


def test_measure_records_duration_even_on_exception() -> None:
    timings = StageTimings()

    try:
        with timings.measure("stage_b"):
            raise ValueError("boom")
    except ValueError:
        pass

    assert "stage_b" in timings.durations_ms


def test_total_ms_sums_all_recorded_stages() -> None:
    timings = StageTimings()
    with timings.measure("a"):
        pass
    with timings.measure("b"):
        pass

    assert timings.total_ms == timings.durations_ms["a"] + timings.durations_ms["b"]
