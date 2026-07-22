"""Prompt D "MAJOR DEDUCTIONS" section: "dangerous_play_review ...
Dangerous-play detection must never automatically disqualify a player. It
must create a review flag."

This is a hand-crafted, fixed-threshold heuristic over Prompt B's kinematic
features -- like `baselines.ThresholdRuleEventDetector` for trick events,
never trained, and deliberately conservative (a wide detection band, so it
flags for human review rather than silently missing anything). It NEVER
outputs a deduction that can affect a score directly: every occurrence it
proposes is a `DeductionType.DANGEROUS_PLAY_REVIEW`, whose ruleset rule sets
`requires_manual_confirmation=True` in the packaged ruleset (see
`rulesets/1a_draft_0_1.yaml`), so by construction (see
`scoring_engine.deduction_is_scorable`) a human must confirm it before it
changes a score at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from yoyovision_ml.domain import DeductionPrediction, DeductionType, FeatureSet
from yoyovision_ml.perception.features import FEATURE_YOYO_VELOCITY

MODEL_NAME = "heuristic-dangerous-play-detector"
MODEL_VERSION = "0.1.0-heuristic"


class DangerousPlayConfig(BaseModel):
    #: Yo-yo velocity (Prompt B `perception.features` units/frame) above
    #: which a frame is treated as a dangerous-speed candidate. Deliberately
    #: a configurable, transparent threshold rather than a learned one.
    velocity_threshold: float = Field(default=8.0, gt=0)
    #: How many consecutive candidate frames must occur before a flag is
    #: raised -- guards against single-frame tracker noise triggering a flag.
    min_consecutive_frames: int = Field(default=3, ge=1)
    #: Candidate runs separated by less than this gap are treated as the
    #: same underlying incident and only flagged once.
    merge_gap_ms: int = Field(default=500, ge=0)


@dataclass(slots=True, frozen=True)
class DangerousPlayFlag:
    """One proposed `dangerous_play_review` occurrence -- always pending
    human confirmation, never auto-applied to a score. See module docstring.
    """

    prediction: DeductionPrediction
    reason: str


def _velocity_series(features: FeatureSet) -> tuple[np.ndarray, np.ndarray]:
    frame_ms = np.array([frame.frame_ms for frame in features.frames], dtype=int)
    velocity = np.array(
        [frame.values.get(FEATURE_YOYO_VELOCITY, 0.0) for frame in features.frames], dtype=float
    )
    return frame_ms, velocity


def detect_dangerous_play(
    features: FeatureSet, config: DangerousPlayConfig | None = None
) -> list[DangerousPlayFlag]:
    """Scans `features` for sustained runs of excessive yo-yo velocity and
    returns one `DangerousPlayFlag` per run (adjacent runs closer than
    `config.merge_gap_ms` apart are merged into a single flag)."""
    config = config or DangerousPlayConfig()
    if not features.frames:
        return []

    frame_ms, velocity = _velocity_series(features)
    is_candidate = velocity >= config.velocity_threshold

    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for i, candidate in enumerate(is_candidate):
        if candidate:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(is_candidate) - 1))

    flags: list[DangerousPlayFlag] = []
    for start_idx, end_idx in runs:
        if (end_idx - start_idx + 1) < config.min_consecutive_frames:
            continue
        start_ms = int(frame_ms[start_idx])
        end_ms = int(frame_ms[end_idx])
        peak_velocity = float(velocity[start_idx : end_idx + 1].max())

        if flags and (start_ms - flags[-1].prediction.timestamp_ms) <= config.merge_gap_ms:
            continue  # treat as a continuation of the same incident

        flags.append(
            DangerousPlayFlag(
                prediction=DeductionPrediction(
                    type=DeductionType.DANGEROUS_PLAY_REVIEW,
                    timestamp_ms=start_ms,
                    quantity=1,
                    confidence=round(
                        min(1.0, peak_velocity / (config.velocity_threshold * 2.0)), 3
                    ),
                    model_name=MODEL_NAME,
                    model_version=MODEL_VERSION,
                ),
                reason=(
                    f"Sustained yo-yo velocity >= {config.velocity_threshold:.1f} between "
                    f"{start_ms}ms-{end_ms}ms (peak {peak_velocity:.1f}); flagged for human "
                    "review -- never automatically scored or used to disqualify a player."
                ),
            )
        )

    return flags
