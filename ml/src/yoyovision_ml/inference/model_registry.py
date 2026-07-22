"""Process-wide model loading: load once, verify checksums first, never
expose local paths.

Prompt F: "Verify model checksum before loading." / "Load models once per
worker process." Today's `adapters_registry.create_*` factories construct a
brand-new adapter (and, for real adapters, reload the checkpoint from disk)
on every call; `pipeline.py` now routes real-adapter construction through
`ModelRegistry.get_or_load` instead so a long-lived Celery worker process
pays that load cost once, not once per job.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from yoyovision_ml.inference.checksums import verify_file_checksum

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ModelArtifactSpec:
    """Identifies one loadable model artefact. `path`/`expected_sha256` are
    `None` for adapters with no on-disk weights (e.g. the mock adapters, or
    MediaPipe's bundled models) -- those skip checksum verification entirely."""

    name: str
    version: str
    path: Path | None = None
    expected_sha256: str | None = None
    device: str = "cpu"


@dataclass(slots=True, frozen=True)
class LoadedModel:
    """A cached adapter instance plus load-time metadata. Never carries the
    original filesystem path in any field that could reach an API response
    or log line surfaced to end users -- see `ModelRegistry.describe`."""

    instance: object
    spec: ModelArtifactSpec
    checksum_sha256: str | None
    load_duration_ms: float
    loaded_at_monotonic: float


class ModelRegistry:
    """Process-wide cache of loaded model adapters, keyed by an explicit
    cache key (adapter kind + name + constructor kwargs), not by object
    identity, so two jobs requesting the same adapter configuration reuse
    the same loaded instance."""

    def __init__(self) -> None:
        self._cache: dict[str, LoadedModel] = {}
        self._lock = Lock()

    def get_or_load(
        self,
        cache_key: str,
        loader: Any,
        *,
        spec: ModelArtifactSpec,
    ) -> LoadedModel:
        """Returns the cached `LoadedModel` for `cache_key`, loading it (and
        verifying its checksum first, if `spec.path` is set) on first use.

        `loader` is a zero-arg callable that constructs the adapter, e.g.
        `lambda: create_yoyo_detector("pytorch", weights_path=..., device=...)`.
        Checksum verification happens *before* `loader()` runs, so a
        tampered/corrupt checkpoint never reaches `torch.load`.
        """
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

            checksum: str | None = None
            if spec.path is not None:
                checksum = verify_file_checksum(
                    spec.path, spec.expected_sha256, label=f"{spec.name}@{spec.version}"
                )

            start = time.monotonic()
            instance = loader()
            load_duration_ms = (time.monotonic() - start) * 1000.0

            loaded = LoadedModel(
                instance=instance,
                spec=spec,
                checksum_sha256=checksum,
                load_duration_ms=load_duration_ms,
                loaded_at_monotonic=start,
            )
            self._cache[cache_key] = loaded
            logger.info(
                "model_loaded",
                extra={
                    "model_name": spec.name,
                    "model_version": spec.version,
                    "device": spec.device,
                    "load_duration_ms": load_duration_ms,
                    "checksum_verified": spec.expected_sha256 is not None,
                },
            )
            return loaded

    def clear(self) -> None:
        """Drops every cached model. Used by tests, and by a worker process
        that needs to force a reload (e.g. after a config change) without
        restarting."""
        with self._lock:
            self._cache.clear()

    def describe(self) -> dict[str, dict[str, object]]:
        """A read-only, path-free summary of what is currently loaded --
        safe to log or expose on a readiness endpoint (Prompt F: "Never
        expose local model paths")."""
        with self._lock:
            return {
                key: {
                    "name": loaded.spec.name,
                    "version": loaded.spec.version,
                    "device": loaded.spec.device,
                    "load_duration_ms": loaded.load_duration_ms,
                    "checksum_verified": loaded.spec.expected_sha256 is not None,
                }
                for key, loaded in self._cache.items()
            }


_default_registry: ModelRegistry | None = None
_default_registry_lock = Lock()


def get_model_registry() -> ModelRegistry:
    """Returns the process-wide `ModelRegistry` singleton, building it on
    first use. One registry per worker process -- exactly the "load once per
    worker process" scope Prompt F asks for."""
    global _default_registry
    if _default_registry is None:
        with _default_registry_lock:
            if _default_registry is None:
                _default_registry = ModelRegistry()
    return _default_registry
