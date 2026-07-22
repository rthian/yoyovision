"""Centralized, env-driven configuration for the workers service.

Deliberately a separate `Settings` class from `yoyovision_api.config` (even
though several env var names overlap) so the two services can be deployed,
versioned, and configured independently -- see package docstring."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"

    database_url: str = (
        "postgresql+asyncpg://yoyovision:yoyovision_dev_password@localhost:5432/yoyovision"
    )

    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    storage_backend: str = "local"
    storage_local_root: str = "/data/storage"

    s3_endpoint_url: str | None = None
    s3_bucket: str = "yoyovision-videos"
    s3_region: str = "us-east-1"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_use_path_style: bool = True
    s3_signed_url_expire_seconds: int = 900

    pipeline_version: str = "0.1.0-dev"
    ruleset_version: str = "1a-draft-0.1"

    #: sampling rate used for frame extraction; kept configurable so a future
    #: GPU-bound detector can trade accuracy for throughput without a code change.
    pipeline_sample_fps: float = 15.0

    #: Prompt F (production inference) settings.
    #: Passed to `run_analysis_pipeline(device_preference=...)`; "cuda" falls
    #: back to "cpu" automatically if unavailable (see `inference.device`).
    pipeline_device: str = "cpu"
    #: Wall-clock budget for one job's pipeline run; exceeding it raises the
    #: (retryable) `PipelineTimeoutError` -- see `inference.cancellation`.
    pipeline_timeout_s: float = 900.0
    #: How often `pipeline_runner` re-reads `analysis_jobs.cancel_requested`
    #: from Postgres while a job is running.
    pipeline_cancel_poll_interval_s: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
