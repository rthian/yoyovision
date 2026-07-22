"""Production-inference concerns for the YoYoVision pipeline (Prompt F).

Everything Prompt A-E built is a *model*; this package is the *runtime*
around those models once they are trusted enough to run inside the worker:
versioned/checksummed model loading, device selection, per-stage timing,
cooperative cancellation, monitoring signals, and human-readable reporting.

Deliberately separate from `pipeline.py`'s orchestration logic so each
concern is independently testable and none of it depends on Celery, FastAPI,
or SQLAlchemy -- `workers/` and `api/` both import from here, never the
other way around.
"""

from __future__ import annotations

from yoyovision_ml.inference.cancellation import CancellationToken
from yoyovision_ml.inference.checksums import compute_file_checksum, verify_file_checksum
from yoyovision_ml.inference.device import DeviceInfo, resolve_device, runtime_versions
from yoyovision_ml.inference.errors import (
    DeterministicPipelineError,
    ModelIntegrityError,
    PipelineCancelledError,
    PipelineTimeoutError,
    TransientPipelineError,
)
from yoyovision_ml.inference.model_registry import (
    LoadedModel,
    ModelArtifactSpec,
    ModelRegistry,
    get_model_registry,
)
from yoyovision_ml.inference.monitoring import (
    MonitoringSignals,
    ReferenceBaseline,
    compute_monitoring_signals,
)
from yoyovision_ml.inference.report import generate_human_readable_report
from yoyovision_ml.inference.timing import StageTimings

__all__ = [
    "CancellationToken",
    "DeterministicPipelineError",
    "DeviceInfo",
    "LoadedModel",
    "ModelArtifactSpec",
    "ModelIntegrityError",
    "ModelRegistry",
    "MonitoringSignals",
    "PipelineCancelledError",
    "PipelineTimeoutError",
    "ReferenceBaseline",
    "StageTimings",
    "TransientPipelineError",
    "compute_file_checksum",
    "compute_monitoring_signals",
    "generate_human_readable_report",
    "get_model_registry",
    "resolve_device",
    "runtime_versions",
    "verify_file_checksum",
]
