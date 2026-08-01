"""Prompt D requirements 7-8: "Support manually entered judge clicks" and
"Support multiple human judge scores." Also the judge-vs-judge half of
requirement 9 ("Calculate agreement and calibration statistics") --
model-vs-judge statistics live in `scoring.calibration`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
from dataclasses import dataclass

from yoyovision_ml.domain import AnalysisEventPrediction, FreestyleEvaluation, Source
from yoyovision_ml.scoring.types import JudgeClick, JudgeFreestyleScore

#: Every automatically-*and*-manually scorable Freestyle Evaluation category,
#: in `domain.FreestyleEvaluation`'s field order (`showmanship` included --
#: judges always enter it manually, but it is still averaged like any other
#: category when multiple judges submit scores).
FE_CATEGORIES: tuple[str, ...] = (
    "execution",
    "control",
    "trick_diversity",
    "space_use_emphasis",
    "music_choreography",
    "music_construction",
    "body_control",
    "showmanship",
)

#: Bland-Altman-style "judges disagree" threshold, in points on the 0-10
#: Freestyle Evaluation scale.
_DISAGREEMENT_THRESHOLD = 3.0

AggregationModeName = Literal["simple_mean", "trim_1", "trim_2", "auto"]
EffectiveAggregationMode = Literal["simple_mean", "trim_1", "trim_2"]


def _resolve_auto_mode(count: int) -> EffectiveAggregationMode:
    if count < 5:
        return "simple_mean"
    if count <= 6:
        return "trim_1"
    return "trim_2"


def _trimmed_mean(values: list[float], drop: int) -> float:
    sorted_values = sorted(values)
    trimmed = sorted_values[drop:-drop] if drop else sorted_values
    return sum(trimmed) / len(trimmed)


def _aggregate_category_values(
    category: str,
    values: list[float],
    mode: AggregationModeName,
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    effective: EffectiveAggregationMode = (
        _resolve_auto_mode(len(values)) if mode == "auto" else mode
    )
    if effective == "simple_mean":
        return sum(values) / len(values), warnings
    if effective == "trim_1":
        if len(values) >= 3:
            return _trimmed_mean(values, 1), warnings
        warnings.append(
            f"trim_1 requested for '{category}' but only {len(values)} value(s); using simple mean."
        )
        return sum(values) / len(values), warnings
    if len(values) >= 5:
        return _trimmed_mean(values, 2), warnings
    warnings.append(
        f"trim_2 requested for '{category}' but only {len(values)} value(s); using simple mean."
    )
    return sum(values) / len(values), warnings


def aggregate_judge_scores(
    scores: Sequence[JudgeFreestyleScore],
    *,
    mode: AggregationModeName = "simple_mean",
) -> tuple[FreestyleEvaluation, list[str]]:
    """Reduces multiple judges' Freestyle Evaluation entries into the single
    `FreestyleEvaluation` `scoring_engine.DeterministicScoringEngine`
    consumes: the per-category mean of whichever judges actually entered
    that category. Warns both when a category has no entries at all, and
    when the judges who did enter it disagree sharply.
    """
    if not scores:
        empty = FreestyleEvaluation(
            execution=None,
            control=None,
            trick_diversity=None,
            space_use_emphasis=None,
            music_choreography=None,
            music_construction=None,
            body_control=None,
            showmanship=None,
            source=Source.HUMAN,
            notes="no judge scores submitted",
        )
        return empty, ["No judge Freestyle Evaluation scores were submitted."]

    warnings: list[str] = []
    aggregated: dict[str, float | None] = {}
    for category in FE_CATEGORIES:
        values = [v for s in scores if (v := getattr(s, category)) is not None]
        if not values:
            aggregated[category] = None
            warnings.append(f"No judge entered a '{category}' score.")
            continue
        category_value, category_warnings = _aggregate_category_values(category, values, mode)
        aggregated[category] = category_value
        warnings.extend(category_warnings)
        if len(values) > 1 and (max(values) - min(values)) >= _DISAGREEMENT_THRESHOLD:
            warnings.append(
                f"Judges disagree on '{category}' (range "
                f"{max(values) - min(values):.1f} points across {len(values)} judges); "
                "the averaged value may be unreliable."
            )

    judge_ids = ", ".join(sorted({s.judge_id for s in scores}))
    evaluation = FreestyleEvaluation(
        **aggregated,
        source=Source.HUMAN,
        notes=f"aggregated ({mode}) across {len(scores)} judge score(s): {judge_ids}",
    )
    return evaluation, warnings


@dataclass(slots=True, frozen=True)
class JudgeAgreement:
    """Per-category absolute point difference between two judges who both
    entered that category -- a simple, direct agreement signal, complementary
    to `scoring.calibration`'s MAE/Pearson/Spearman/ICC (which are meant to
    run over *many* routines, not a single pair of scores)."""

    judge_a: str
    judge_b: str
    category: str
    absolute_difference: float


def pairwise_judge_agreement(scores: Sequence[JudgeFreestyleScore]) -> list[JudgeAgreement]:
    results: list[JudgeAgreement] = []
    for i, judge_a in enumerate(scores):
        for judge_b in scores[i + 1 :]:
            if judge_a.judge_id == judge_b.judge_id:
                continue
            for category in FE_CATEGORIES:
                value_a, value_b = getattr(judge_a, category), getattr(judge_b, category)
                if value_a is None or value_b is None:
                    continue
                results.append(
                    JudgeAgreement(
                        judge_a=judge_a.judge_id,
                        judge_b=judge_b.judge_id,
                        category=category,
                        absolute_difference=abs(value_a - value_b),
                    )
                )
    return results


@dataclass(slots=True, frozen=True)
class ClickMatch:
    """Whether/how well one `JudgeClick` lines up with a detected event."""

    click: JudgeClick
    matched_event_label: str | None
    #: model event start_ms - click timestamp_ms (signed; None if unmatched).
    boundary_error_ms: int | None


def match_clicks_to_events(
    clicks: Sequence[JudgeClick],
    events: Sequence[AnalysisEventPrediction],
    tolerance_ms: int = 1000,
) -> list[ClickMatch]:
    """For each judge click, finds the closest model event start within
    `tolerance_ms` (restricted to `click.associated_label` when the judge
    entered one). Directly answers `dataset.schema.JudgeClickAnnotation`'s
    stated purpose: "check how closely model-detected event boundaries track
    a judge's real-time perception" -- see `scoring.calibration.event_count_agreement`
    for the precision/recall summary built on top of this.
    """
    matches: list[ClickMatch] = []
    for click in clicks:
        candidates = [
            event
            for event in events
            if click.associated_label is None or event.label == click.associated_label
        ]
        best_event: AnalysisEventPrediction | None = None
        best_distance: int | None = None
        for event in candidates:
            distance = abs(event.start_ms - click.timestamp_ms)
            if distance <= tolerance_ms and (best_distance is None or distance < best_distance):
                best_event, best_distance = event, distance
        if best_event is None:
            matches.append(
                ClickMatch(click=click, matched_event_label=None, boundary_error_ms=None)
            )
        else:
            matches.append(
                ClickMatch(
                    click=click,
                    matched_event_label=best_event.label,
                    boundary_error_ms=best_event.start_ms - click.timestamp_ms,
                )
            )
    return matches
