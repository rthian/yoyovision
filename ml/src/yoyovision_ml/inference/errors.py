"""Failure taxonomy for production inference.

Prompt F: "Retry only transient failures. Never retry deterministic
validation failures." Everything raised by pipeline/worker code during a
job run must be one of these two families so `workers/tasks.py` can decide,
mechanically, whether a Celery retry is safe -- it must never guess based on
exception message text.
"""

from __future__ import annotations


class TransientPipelineError(Exception):
    """A failure that may succeed on retry: storage/network hiccups, a busy
    GPU, a broker/DB blip, or a soft timeout that might pass with less
    contention next time. Celery tasks retry these automatically."""


class PipelineTimeoutError(TransientPipelineError):
    """Raised when a job exceeds its deadline (see `CancellationToken`).
    Treated as transient: the same job may finish in time when the worker
    is less loaded."""


class DeterministicPipelineError(Exception):
    """A failure that will not change on retry: malformed input, a corrupt
    or tampered model checkpoint, or an explicit cancellation. Retrying
    wastes worker capacity and never resolves the job, so these are never
    retried -- they always mark the job `failed` immediately."""


class ModelIntegrityError(DeterministicPipelineError):
    """A model artefact's checksum did not match the expected value, or no
    expected checksum was configured for a required artefact. Never retried
    -- the on-disk file will not change between attempts."""


class PipelineCancelledError(DeterministicPipelineError):
    """Raised when a human explicitly cancels a running job. Retrying would
    contradict the human's request, so this is deterministic, not transient."""
