"""Builds a `StoragePort` from worker `Settings`.

Mirrors `yoyovision_api.deps.get_storage`'s backend-selection logic, but is
defined independently here (no import of `yoyovision_api`) since storage
backend construction is plain configuration, not shared business logic --
see the `yoyovision_workers` package docstring on why the two services don't
share Python code."""

from __future__ import annotations

# Import for side-effect: registers "local"/"s3" storage backends with the ml registry.
from yoyovision_ml import storage as _storage_adapters  # noqa: F401
from yoyovision_ml.adapters_registry import create_storage_backend
from yoyovision_ml.interfaces import StoragePort

from yoyovision_workers.config import Settings


def build_storage(settings: Settings) -> StoragePort:
    if settings.storage_backend == "local":
        backend = create_storage_backend("local", root=settings.storage_local_root)
    elif settings.storage_backend == "s3":
        backend = create_storage_backend(
            "s3",
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            use_path_style=settings.s3_use_path_style,
        )
    else:
        raise ValueError(f"Unknown STORAGE_BACKEND: {settings.storage_backend!r}")
    return backend  # type: ignore[return-value]
