"""Scoring profiles and judge calibration (Prompt D).

Extends the existing deterministic `yoyovision_ml.scoring_engine`/
`yoyovision_ml.ruleset` (which remain the single source of truth for the
actual point math) with everything Prompt D adds on top:

- `profiles`: multiple named scoring profiles (`practice`/`judge_assist`/`research`).
- `types`: judge clicks, multi-judge Freestyle Evaluation scores, per-event
  manual overrides, automatic Freestyle-Evaluation estimates, and the
  multi-stage `ScoringPipelineResult`.
- `overrides`: applies per-event manual overrides with an audit trail.
- `judges`: aggregates multiple judges' Freestyle Evaluation scores and
  matches judge clicks to detected events.
- `fe_estimators`: optional, hand-crafted automatic estimators for 6 of the
  8 Freestyle Evaluation categories (never `showmanship`).
- `dangerous_play`: a hand-crafted, review-only `dangerous_play_review`
  flag detector -- never auto-applied to a score.
- `pipeline`: the profile-aware orchestrator tying every stage together
  into one auditable `ScoringPipelineResult`.
- `calibration`: MAE/Pearson/Spearman/ICC/event-count precision-recall/
  Bland-Altman statistics, plus an optional calibration plot.
- `cli`: the `yoyovision-scoring` console script.

Deliberately kept `ml`-only for now (no new API routes, DB tables, or
frontend UI) -- mirrors how Prompt C's `events/` package shipped a fully
working, tested model before any pipeline integration. See
`docs/adapters.md`'s "Scoring & judge calibration" section for the current
scope and what is intentionally deferred.
"""
