"""Tests for `yoyovision_ml.inference.checksums`."""

from __future__ import annotations

from pathlib import Path

import pytest
from yoyovision_ml.inference.checksums import compute_file_checksum, verify_file_checksum
from yoyovision_ml.inference.errors import ModelIntegrityError


def test_compute_file_checksum_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"pretend-model-weights")

    assert compute_file_checksum(path) == compute_file_checksum(path)


def test_compute_file_checksum_changes_with_content(tmp_path: Path) -> None:
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    path_a.write_bytes(b"one")
    path_b.write_bytes(b"two")

    assert compute_file_checksum(path_a) != compute_file_checksum(path_b)


def test_verify_file_checksum_passes_when_matching(tmp_path: Path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"trusted-checkpoint")
    expected = compute_file_checksum(path)

    actual = verify_file_checksum(path, expected, label="test-model")

    assert actual == expected


def test_verify_file_checksum_raises_on_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"tampered-checkpoint")

    with pytest.raises(ModelIntegrityError, match="Checksum mismatch"):
        verify_file_checksum(path, "0" * 64, label="test-model")


def test_verify_file_checksum_raises_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.bin"

    with pytest.raises(ModelIntegrityError, match="does not exist"):
        verify_file_checksum(missing, None, label="test-model")


def test_verify_file_checksum_skips_verification_when_no_expected_checksum(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unpinned.bin"
    path.write_bytes(b"no-checksum-pinned-yet")

    # Should not raise even though nothing was pinned -- just logs a warning.
    actual = verify_file_checksum(path, None, label="unpinned-model")

    assert actual == compute_file_checksum(path)
