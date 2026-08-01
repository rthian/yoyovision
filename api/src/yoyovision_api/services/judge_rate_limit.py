"""Best-effort in-process rate limiting for judge invite resolution."""

from __future__ import annotations

import time
from collections import defaultdict, deque

_WINDOW_SECONDS = 60.0
_buckets: dict[str, deque[float]] = defaultdict(deque)


class JudgeRateLimitExceeded(Exception):
    """Too many judge-access requests for this key."""


def check_judge_rate_limit(key: str, *, limit_per_minute: int) -> None:
    now = time.monotonic()
    bucket = _buckets[key]
    while bucket and now - bucket[0] > _WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= limit_per_minute:
        raise JudgeRateLimitExceeded("Too many requests. Try again shortly.")
    bucket.append(now)
