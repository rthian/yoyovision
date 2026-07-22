"""Model-artefact checksum verification.

Prompt F: "Verify model checksum before loading." Distinct from
`perception.artifact.compute_video_checksum` (which fingerprints an *input*
video for provenance) -- this fingerprints *model weight files* before they
are loaded into memory, so a corrupted or tampered checkpoint fails loudly
instead of silently producing garbage predictions. Today no real weight
files ship with this repository (see README's "Current model status"), so
this is exercised by `model_registry.py` against whatever checkpoint path is
configured, with an explicit `expected_sha256=None` fallback that only warns
(see `verify_file_checksum`'s docstring) rather than blocking development
before any checksum has been pinned.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from yoyovision_ml.inference.errors import ModelIntegrityError

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB


def compute_file_checksum(path: Path, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> str:
    """Streaming SHA-256 of `path`. Never loads the whole file into memory,
    since model checkpoints can be large."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_checksum(
    path: Path,
    expected_sha256: str | None,
    *,
    label: str,
) -> str:
    """Verifies `path` matches `expected_sha256` before it is loaded.

    Raises `ModelIntegrityError` (deterministic -- never retried) if the
    file is missing or the checksum does not match. If `expected_sha256` is
    `None` (no checksum has been pinned for this artefact yet), verification
    is skipped but a warning is logged -- this keeps local development
    working before a real checkpoint's checksum has been recorded, while
    still making the gap visible rather than silent. Returns the computed
    checksum either way, so callers can persist/log it.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise ModelIntegrityError(f"Model artefact for '{label}' does not exist: {resolved}")

    actual = compute_file_checksum(resolved)
    if expected_sha256 is None:
        logger.warning(
            "model_checksum_unpinned",
            extra={"label": label, "actual_sha256": actual},
        )
        return actual

    if actual != expected_sha256:
        raise ModelIntegrityError(
            f"Checksum mismatch for '{label}': expected {expected_sha256}, got {actual}. "
            "Refusing to load a model artefact that does not match its pinned checksum."
        )
    return actual
