"""Cooperative cancellation and timeout handling.

Prompt F: "Add cancellation and timeout handling." The pipeline is a plain
synchronous function with no thread/process boundary of its own, so
cancellation cannot be pre-emptive -- `CancellationToken.check(...)` is
polled at each `PipelineStage` boundary in `pipeline.py` and raises
promptly instead of running the (potentially expensive) next stage.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from yoyovision_ml.inference.errors import PipelineCancelledError, PipelineTimeoutError


@dataclass(slots=True)
class CancellationToken:
    """Polled cooperatively between pipeline stages.

    `cancel_check`: an optional zero-arg callable a caller (the worker) can
    wire up to re-read a "cancel requested" flag (e.g. from the DB) without
    this module knowing anything about Celery or SQLAlchemy.

    `timeout_s`: an optional wall-clock budget for the whole pipeline run,
    measured from construction time via `time.monotonic()` (never wall-clock
    `time.time()`, which can jump backwards across clock adjustments).
    """

    cancel_check: Callable[[], bool] | None = None
    timeout_s: float | None = None
    _deadline: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._deadline = time.monotonic() + self.timeout_s if self.timeout_s is not None else None

    def check(self, stage_name: str) -> None:
        """Raises `PipelineCancelledError` or `PipelineTimeoutError` if this
        token has been cancelled or its deadline has passed. Called at every
        stage boundary; a no-op token (no `cancel_check`/`timeout_s`) never
        raises."""
        if self._deadline is not None and time.monotonic() > self._deadline:
            raise PipelineTimeoutError(
                f"Pipeline exceeded its time budget before stage '{stage_name}'."
            )
        if self.cancel_check is not None and self.cancel_check():
            raise PipelineCancelledError(f"Pipeline was cancelled before stage '{stage_name}'.")

    @property
    def remaining_s(self) -> float | None:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.monotonic())
