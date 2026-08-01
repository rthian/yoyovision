"""Compute multi-judge panel results with aggregation and AI profiles."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from yoyovision_api.db_models import (
    FreestyleEvaluationORM,
    JudgeAssignmentORM,
    JudgeFreestyleScoreORM,
    JudgingEntryORM,
    JudgingEntryVideoORM,
)
from yoyovision_api.judging_enums import AiMixProfile, AggregationMode
from yoyovision_api.schemas import (
    FeCategoryScores,
    JudgeResultRow,
    JudgingEntryResultsRead,
    VideoResults,
)
from yoyovision_api.services.judging_service import JudgingServiceError
from yoyovision_ml.scoring.judges import FE_CATEGORIES, aggregate_judge_scores
from yoyovision_ml.scoring.types import JudgeFreestyleScore

AI_JUDGE_ID = "__ai__"


def _scores_from_fe(fe: FreestyleEvaluationORM | None) -> FeCategoryScores | None:
    if fe is None:
        return None
    return FeCategoryScores(
        execution=fe.execution,
        control=fe.control,
        trick_diversity=fe.trick_diversity,
        space_use_emphasis=fe.space_use_emphasis,
        music_choreography=fe.music_choreography,
        music_construction=fe.music_construction,
        body_control=fe.body_control,
        showmanship=fe.showmanship,
    )


def _scores_from_evaluation(evaluation) -> FeCategoryScores:
    return FeCategoryScores(
        execution=evaluation.execution,
        control=evaluation.control,
        trick_diversity=evaluation.trick_diversity,
        space_use_emphasis=evaluation.space_use_emphasis,
        music_choreography=evaluation.music_choreography,
        music_construction=evaluation.music_construction,
        body_control=evaluation.body_control,
        showmanship=evaluation.showmanship,
    )


def _orm_to_ml_score(assignment: JudgeAssignmentORM, score: JudgeFreestyleScoreORM) -> JudgeFreestyleScore:
    return JudgeFreestyleScore(
        judge_id=assignment.id,
        execution=score.execution,
        control=score.control,
        trick_diversity=score.trick_diversity,
        space_use_emphasis=score.space_use_emphasis,
        music_choreography=score.music_choreography,
        music_construction=score.music_construction,
        body_control=score.body_control,
        showmanship=score.showmanship,
        notes=score.notes,
    )


def _ml_score_from_fe(scores: FeCategoryScores, *, judge_id: str) -> JudgeFreestyleScore:
    return JudgeFreestyleScore(
        judge_id=judge_id,
        execution=scores.execution,
        control=scores.control,
        trick_diversity=scores.trick_diversity,
        space_use_emphasis=scores.space_use_emphasis,
        music_choreography=scores.music_choreography,
        music_construction=scores.music_construction,
        body_control=scores.body_control,
        showmanship=scores.showmanship,
        notes="ai virtual judge",
    )


def _aggregation_mode_value(mode: AggregationMode) -> str:
    return mode.value


def _apply_profile_b(
    human: FeCategoryScores, ai_fe: FeCategoryScores | None
) -> tuple[FeCategoryScores, list[str]]:
    if ai_fe is None:
        return human, []
    filled: list[str] = []
    merged = human.model_copy()
    for category in FE_CATEGORIES:
        human_value = getattr(merged, category)
        ai_value = getattr(ai_fe, category)
        if human_value is None and ai_value is not None:
            setattr(merged, category, ai_value)
            filled.append(category)
    return merged, filled


def _apply_profile_c(
    human_scores: list[JudgeFreestyleScore],
    ai_fe: FeCategoryScores | None,
    mode: AggregationMode,
) -> tuple[FeCategoryScores, bool, list[str]]:
    warnings: list[str] = []
    if ai_fe is None or not any(getattr(ai_fe, category) is not None for category in FE_CATEGORIES):
        evaluation, agg_warnings = aggregate_judge_scores(
            human_scores, mode=_aggregation_mode_value(mode)
        )
        warnings.extend(agg_warnings)
        return _scores_from_evaluation(evaluation), False, warnings

    combined = [*human_scores, _ml_score_from_fe(ai_fe, judge_id=AI_JUDGE_ID)]
    evaluation, agg_warnings = aggregate_judge_scores(
        combined, mode=_aggregation_mode_value(mode)
    )
    warnings.extend(agg_warnings)
    return _scores_from_evaluation(evaluation), True, warnings


async def _load_fe_map(
    session: AsyncSession, analysis_ids: set[str]
) -> dict[str, FreestyleEvaluationORM]:
    if not analysis_ids:
        return {}
    result = await session.execute(
        select(FreestyleEvaluationORM).where(
            FreestyleEvaluationORM.analysis_id.in_(analysis_ids)
        )
    )
    return {row.analysis_id: row for row in result.scalars().all()}


async def compute_entry_results(session: AsyncSession, entry_id: str) -> JudgingEntryResultsRead:
    result = await session.execute(
        select(JudgingEntryORM)
        .where(JudgingEntryORM.id == entry_id)
        .options(
            selectinload(JudgingEntryORM.videos).selectinload(JudgingEntryVideoORM.video),
            selectinload(JudgingEntryORM.judges).selectinload(JudgeAssignmentORM.freestyle_scores),
        )
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        raise JudgingServiceError("Judging entry not found.")

    analysis_ids: set[str] = set()
    for video in entry.videos:
        if video.official_analysis_id:
            analysis_ids.add(video.official_analysis_id)
        if video.shadow_analysis_id:
            analysis_ids.add(video.shadow_analysis_id)
    fe_by_analysis = await _load_fe_map(session, analysis_ids)

    entry_warnings: list[str] = []
    if not entry.judges:
        entry_warnings.append("No judges assigned to this entry.")

    video_results: list[VideoResults] = []
    for entry_video in sorted(entry.videos, key=lambda row: row.sort_order):
        video_warnings: list[str] = []
        judge_rows: list[JudgeResultRow] = []
        aggregate_pool: list[JudgeFreestyleScore] = []

        for assignment in entry.judges:
            score = next(
                (row for row in assignment.freestyle_scores if row.entry_video_id == entry_video.id),
                None,
            )
            if score is None:
                continue
            included = (
                score.is_submitted
                and assignment.include_in_results
                and not assignment.is_shadow
            )
            if included:
                aggregate_pool.append(_orm_to_ml_score(assignment, score))
            judge_rows.append(
                JudgeResultRow(
                    assignment_id=assignment.id,
                    display_name=assignment.display_name,
                    include_in_results=assignment.include_in_results,
                    is_shadow=assignment.is_shadow,
                    is_submitted=score.is_submitted,
                    included_in_aggregate=included,
                    scores=FeCategoryScores(
                        execution=score.execution,
                        control=score.control,
                        trick_diversity=score.trick_diversity,
                        space_use_emphasis=score.space_use_emphasis,
                        music_choreography=score.music_choreography,
                        music_construction=score.music_construction,
                        body_control=score.body_control,
                        showmanship=score.showmanship,
                    ),
                    notes=score.notes,
                )
            )

        mode_value = _aggregation_mode_value(entry.aggregation_mode)
        if aggregate_pool:
            human_eval, agg_warnings = aggregate_judge_scores(aggregate_pool, mode=mode_value)
            human_aggregate = _scores_from_evaluation(human_eval)
            video_warnings.extend(agg_warnings)
        else:
            human_aggregate = FeCategoryScores()
            video_warnings.append("No submitted judge scores included in aggregation.")

        ai_fe = (
            _scores_from_fe(fe_by_analysis.get(entry_video.official_analysis_id))
            if entry_video.official_analysis_id
            else None
        )
        shadow_fe = (
            _scores_from_fe(fe_by_analysis.get(entry_video.shadow_analysis_id))
            if entry_video.shadow_analysis_id
            else None
        )
        if entry_video.official_analysis_id and ai_fe is None:
            video_warnings.append(
                f"No Freestyle Evaluation on official analysis {entry_video.official_analysis_id}."
            )
        if entry_video.shadow_analysis_id and shadow_fe is None:
            video_warnings.append(
                f"No Freestyle Evaluation on shadow analysis {entry_video.shadow_analysis_id}."
            )

        ai_filled: list[str] = []
        ai_virtual = False
        if entry.ai_mix_profile == AiMixProfile.COMPARE_ONLY:
            panel = human_aggregate
        elif entry.ai_mix_profile == AiMixProfile.GAP_FILL:
            panel, ai_filled = _apply_profile_b(human_aggregate, ai_fe)
        else:
            panel, ai_virtual, profile_warnings = _apply_profile_c(
                aggregate_pool, ai_fe, entry.aggregation_mode
            )
            video_warnings.extend(profile_warnings)

        video_results.append(
            VideoResults(
                entry_video_id=entry_video.id,
                video_id=entry_video.video_id,
                sort_order=entry_video.sort_order,
                original_filename=entry_video.video.original_filename,
                official_analysis_id=entry_video.official_analysis_id,
                shadow_analysis_id=entry_video.shadow_analysis_id,
                judges=judge_rows,
                panel_aggregate=panel,
                human_aggregate=human_aggregate,
                ai_fe=ai_fe,
                shadow_fe=shadow_fe,
                ai_filled_categories=ai_filled,
                ai_virtual_judge_included=ai_virtual,
                effective_aggregation_mode=mode_value,
                warnings=video_warnings,
            )
        )

    return JudgingEntryResultsRead(
        entry_id=entry.id,
        title=entry.title,
        mode=entry.mode,
        status=entry.status,
        ai_mix_profile=entry.ai_mix_profile,
        aggregation_mode=entry.aggregation_mode,
        videos=video_results,
        warnings=entry_warnings,
    )
