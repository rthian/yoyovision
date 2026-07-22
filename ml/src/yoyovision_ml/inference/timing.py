"""Per-stage duration recording.

Prompt F: "Record inference duration per stage." `pipeline.py` wraps each
`PipelineStage` in `StageTimings.measure(...)` so `PipelineResult` always
carries a `stage_durations_ms` mapping, independent of whether real or mock
adapters ran.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class StageTimings:
    durations_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, stage_name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations_ms[stage_name] = (time.perf_counter() - start) * 1000.0

    @property
    def total_ms(self) -> float:
        return sum(self.durations_ms.values())
