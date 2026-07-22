"""Storage adapters implementing `StoragePort`.

Two interchangeable backends: a local-filesystem adapter for development and
an S3-compatible adapter for production (also usable locally against MinIO).
Selection happens via `STORAGE_BACKEND` config + `adapters_registry`, never
via direct imports in calling code, so swapping backends never touches
routers/services.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from yoyovision_ml.adapters_registry import register_storage_backend


class StorageKeyTraversalError(ValueError):
    """Raised when a storage key attempts path traversal outside the storage root."""


def _assert_safe_relative_key(storage_key: str) -> None:
    if not storage_key or storage_key.startswith("/") or storage_key.startswith("\\"):
        raise StorageKeyTraversalError(f"Storage key must be a safe relative path: {storage_key!r}")
    normalized = os.path.normpath(storage_key)
    if normalized.startswith("..") or normalized.startswith(f"..{os.sep}"):
        raise StorageKeyTraversalError(f"Storage key attempts path traversal: {storage_key!r}")


@register_storage_backend("local")
class LocalFilesystemStorage:
    """Development storage backend. Never exposes real filesystem paths to clients."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, storage_key: str) -> Path:
        _assert_safe_relative_key(storage_key)
        target = (self._root / storage_key).resolve()
        if self._root not in target.parents and target != self._root:
            raise StorageKeyTraversalError(f"Resolved path escapes storage root: {storage_key!r}")
        return target

    def put(self, storage_key: str, data: bytes, content_type: str) -> None:
        path = self._resolve(storage_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, storage_key: str) -> bytes:
        return self._resolve(storage_key).read_bytes()

    def delete(self, storage_key: str) -> None:
        path = self._resolve(storage_key)
        if path.exists():
            path.unlink()

    def exists(self, storage_key: str) -> bool:
        return self._resolve(storage_key).exists()

    def signed_url(self, storage_key: str, expires_seconds: int) -> str:
        # Local dev has no real signed-URL mechanism; callers must proxy
        # downloads through an authenticated API endpoint instead. This
        # deliberately does not expose a filesystem path to any client.
        _assert_safe_relative_key(storage_key)
        return f"/api/videos/download-proxy?key={storage_key}&expires_in={expires_seconds}"


@register_storage_backend("s3")
class S3CompatibleStorage:
    """Production storage backend (also used locally against MinIO for parity)."""

    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        use_path_style: bool = True,
    ) -> None:
        try:
            import boto3
            from botocore.client import Config
        except ImportError as exc:  # pragma: no cover - exercised only without boto3 installed
            raise RuntimeError(
                "boto3 is required for the 's3' storage backend. Install it via "
                "`pip install boto3` or use STORAGE_BACKEND=local."
            ) from exc

        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(s3={"addressing_style": "path" if use_path_style else "auto"}),
        )

    def put(self, storage_key: str, data: bytes, content_type: str) -> None:
        _assert_safe_relative_key(storage_key)
        self._client.put_object(
            Bucket=self._bucket, Key=storage_key, Body=data, ContentType=content_type
        )

    def get(self, storage_key: str) -> bytes:
        _assert_safe_relative_key(storage_key)
        response = self._client.get_object(Bucket=self._bucket, Key=storage_key)
        return cast(bytes, response["Body"].read())

    def delete(self, storage_key: str) -> None:
        _assert_safe_relative_key(storage_key)
        self._client.delete_object(Bucket=self._bucket, Key=storage_key)

    def exists(self, storage_key: str) -> bool:
        _assert_safe_relative_key(storage_key)
        try:
            self._client.head_object(Bucket=self._bucket, Key=storage_key)
            return True
        except Exception:
            return False

    def signed_url(self, storage_key: str, expires_seconds: int) -> str:
        _assert_safe_relative_key(storage_key)
        return cast(
            str,
            self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": storage_key},
                ExpiresIn=expires_seconds,
            ),
        )
