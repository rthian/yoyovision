"""Centralized, env-driven configuration. No secrets are hard-coded here;
all values come from environment variables (see .env.example)."""

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

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    auth_jwt_secret: str = "change-me-dev-only-secret"
    auth_jwt_algorithm: str = "HS256"
    auth_jwt_expire_minutes: int = 1440
    auth_dev_seed_user_email: str = "dev@yoyovision.local"
    auth_dev_seed_user_password: str = "change-me-dev-only-password"

    storage_backend: str = "local"
    storage_local_root: str = "/data/storage"
    storage_max_upload_bytes: int = 524_288_000
    storage_max_duration_ms: int = 600_000

    s3_endpoint_url: str | None = None
    s3_bucket: str = "yoyovision-videos"
    s3_region: str = "us-east-1"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_use_path_style: bool = True
    s3_signed_url_expire_seconds: int = 900

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: str = "http://localhost:3000"
    api_rate_limit_per_minute: int = 60

    pipeline_version: str = "0.1.0-dev"
    ruleset_version: str = "1a-draft-0.1"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.api_cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
