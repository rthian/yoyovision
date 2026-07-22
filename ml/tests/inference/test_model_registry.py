"""Tests for `yoyovision_ml.inference.model_registry`."""

from __future__ import annotations

from pathlib import Path

import pytest
from yoyovision_ml.inference.checksums import compute_file_checksum
from yoyovision_ml.inference.errors import ModelIntegrityError
from yoyovision_ml.inference.model_registry import (
    ModelArtifactSpec,
    ModelRegistry,
    get_model_registry,
)


def test_get_or_load_loads_once_and_caches() -> None:
    registry = ModelRegistry()
    calls = {"count": 0}

    def loader() -> object:
        calls["count"] += 1
        return object()

    spec = ModelArtifactSpec(name="dummy", version="v1")
    first = registry.get_or_load("dummy-key", loader, spec=spec)
    second = registry.get_or_load("dummy-key", loader, spec=spec)

    assert calls["count"] == 1
    assert first.instance is second.instance


def test_get_or_load_different_keys_load_independently() -> None:
    registry = ModelRegistry()
    calls = {"count": 0}

    def loader() -> object:
        calls["count"] += 1
        return object()

    spec = ModelArtifactSpec(name="dummy", version="v1")
    registry.get_or_load("key-a", loader, spec=spec)
    registry.get_or_load("key-b", loader, spec=spec)

    assert calls["count"] == 2


def test_get_or_load_verifies_checksum_before_loading(tmp_path: Path) -> None:
    registry = ModelRegistry()
    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"real-checkpoint-bytes")
    expected = compute_file_checksum(weights_path)

    spec = ModelArtifactSpec(
        name="yoyo-detector", version="v1", path=weights_path, expected_sha256=expected
    )
    loaded = registry.get_or_load("yoyo-key", lambda: "loaded-adapter", spec=spec)

    assert loaded.instance == "loaded-adapter"
    assert loaded.checksum_sha256 == expected


def test_get_or_load_raises_and_never_loads_on_checksum_mismatch(tmp_path: Path) -> None:
    registry = ModelRegistry()
    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"tampered-checkpoint-bytes")
    calls = {"count": 0}

    def loader() -> object:
        calls["count"] += 1
        return object()

    spec = ModelArtifactSpec(
        name="yoyo-detector", version="v1", path=weights_path, expected_sha256="0" * 64
    )

    with pytest.raises(ModelIntegrityError):
        registry.get_or_load("yoyo-key", loader, spec=spec)

    assert calls["count"] == 0, "loader must never run when the checksum check fails"


def test_describe_never_exposes_the_local_path(tmp_path: Path) -> None:
    registry = ModelRegistry()
    weights_path = tmp_path / "secret-local-weights.bin"
    weights_path.write_bytes(b"weights")

    spec = ModelArtifactSpec(name="yoyo-detector", version="v1", path=weights_path)
    registry.get_or_load("yoyo-key", lambda: "adapter", spec=spec)

    description = registry.describe()
    assert str(weights_path) not in repr(description)
    assert description["yoyo-key"]["checksum_verified"] is False  # expected_sha256 was None


def test_clear_forces_a_reload() -> None:
    registry = ModelRegistry()
    calls = {"count": 0}

    def loader() -> object:
        calls["count"] += 1
        return object()

    spec = ModelArtifactSpec(name="dummy", version="v1")
    registry.get_or_load("dummy-key", loader, spec=spec)
    registry.clear()
    registry.get_or_load("dummy-key", loader, spec=spec)

    assert calls["count"] == 2


def test_get_model_registry_returns_a_process_wide_singleton() -> None:
    assert get_model_registry() is get_model_registry()
