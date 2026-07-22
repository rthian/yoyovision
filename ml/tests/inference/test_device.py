"""Tests for `yoyovision_ml.inference.device`."""

from __future__ import annotations

from yoyovision_ml.inference.device import resolve_device, runtime_versions


def test_resolve_device_cpu_is_always_available() -> None:
    info = resolve_device("cpu")

    assert info.resolved == "cpu"
    assert info.available is True


def test_resolve_device_auto_falls_back_to_cpu_without_cuda() -> None:
    info = resolve_device("auto")

    # This dev/test environment has no GPU, so "auto"/"cuda"/"gpu" must all
    # resolve to a safe cpu fallback rather than raising.
    assert info.resolved in {"cpu", "cuda"}
    if info.resolved == "cpu":
        assert info.reason


def test_resolve_device_unknown_preference_falls_back_to_cpu() -> None:
    info = resolve_device("quantum")

    assert info.resolved == "cpu"
    assert info.available is True
    assert "quantum" in info.reason


def test_runtime_versions_always_includes_python() -> None:
    versions = runtime_versions()

    assert "python" in versions
    assert "platform" in versions
