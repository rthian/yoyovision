# Ruleset & Scoring

## ⚠️ Disclaimer

**YoYoVision is a training and judge-assistance tool. No score it produces
is, or ever will be claimed to be, an officially certified score of IYYF,
WYYC, or any other competition body.** The packaged ruleset
(`1a-draft-0.1`, `ml/src/yoyovision_ml/rulesets/1a_draft_0_1.yaml`) is an
**unofficial first draft** authored for this application only, modeled
loosely on the publicly known *structure* of 1A freestyle judging
(technical elements, major deductions, freestyle categories). It is **not**
sourced from, endorsed by, or an official published trick-value table of any
organization. This disclaimer is not just documentation — it is enforced in
code:

- `Ruleset.is_official` defaults to `False` and `1a_draft_0_1.yaml` sets it
  explicitly.
- `Ruleset.disclaimer` is a required-by-default string field, surfaced
  verbatim by the `/rulesets` API endpoints and the frontend's Ruleset
  transparency panel.
- `DeterministicScoringEngine.calculate()` always injects an
  `_UNOFFICIAL_WARNING` string into `ScoreBreakdown.warnings`, plus a
  second warning quoting `ruleset.disclaimer` whenever `is_official=False`
  — i.e. every score this engine has ever produced carries the disclaimer,
  not just the UI copy around it.

Any organization that wants to use YoYoVision against a real published
rulebook must supply and version their own ruleset YAML file and flip
`is_official: true` themselves; the platform never does this for them.

## Why rules-based, not learned scoring

Product principle #1 forbids an opaque model that predicts a final score
directly. YoYoVision's pipeline instead produces a list of already-detected,
timestamped, evidenced `AnalysisEventPrediction`/`DeductionPrediction`
objects (see `docs/data_model.md`), and a separate, swappable
`ScoringEngine.calculate(events, deductions, freestyle_evaluation, ruleset)`
step (`interfaces.py`) turns those into a `ScoreBreakdown`. The only shipped
implementation, `DeterministicScoringEngine`
(`ml/src/yoyovision_ml/scoring_engine.py`), does pure arithmetic over a
`Ruleset` — no model weights, no learned parameters, fully reproducible and
auditable from the event list alone.

## Ruleset schema (`ruleset.py`)

| Field | Type | Meaning |
| --- | --- | --- |
| `version` | `str` | Free-form version string (e.g. `"1a-draft-0.1"`). Stamped onto every `ScoreBreakdown.ruleset_version` produced with it. |
| `is_official` | `bool` (default `False`) | Whether this ruleset represents an officially sanctioned rulebook. Always `False` for the packaged draft. |
| `iyyf_certification_reference` | `str \| None` | Prompt D requirement 4 ("Never label a custom trick difficulty value as an official IYYF value"), enforced structurally: a model validator rejects `is_official=True` unless this cites a concrete certification source. No packaged ruleset sets either field. |
| `disclaimer` | `str` | Human-readable disclaimer text, always surfaced with any score computed under this ruleset. |
| `difficulty_band_points` | `DifficultyBandPoints` | Base points per `basic`/`intermediate`/`advanced`/`unknown` `DifficultyBand`. **Not** an official difficulty rating — see below. |
| `repeated_element_decay` | `RepeatedElementDecay` | Reduced-credit curve for repeated elements, plus which `policy` (`decay_high_risk_only` (default) / `decay_all_families` / `full_credit` / `cap_occurrences`) governs whether that curve applies only to a configurable set of `high_risk_families`, to every family, not at all, or with a hard repeat cap — see "Repeated-element policies" below. |
| `deduction_rules` | `list[DeductionRule]` | Points-per-occurrence, an optional penalty cap, and a `requires_manual_confirmation` flag (Prompt D — see "Dangerous-play review flags" below), one entry per `DeductionType`. |
| `freestyle_evaluation_weights` | `FreestyleEvaluationWeights` | Per-category weight (default `1.0` each) for the 8 manual Freestyle Evaluation categories. |
| `technical_scale_max` / `freestyle_evaluation_scale_max` | `float` | Ceilings the raw technical/FE totals are scaled/clamped to before blending. |
| `technical_weight` / `freestyle_evaluation_weight` | `float` | Blend weights for the final score; validated to sum to exactly `1.0` (`_weights_sum_to_one`). |
| `low_confidence_review_threshold` | `float` (default `0.55`) | Any event/deduction below this confidence contributes a "requires human review" warning (product principle #3). |

`load_ruleset(path)` parses and validates a YAML file into this model;
`default_ruleset()` (LRU-cached) loads the packaged `1a-draft-0.1`;
`list_available_rulesets()` / `get_ruleset_by_version()` support the
`/rulesets` transparency endpoints, which let a user inspect the *exact*
versioned configuration that produced any given score.

### Important: difficulty bands are model output, not an official value table

`AnalysisEvent.difficulty_band` is assigned by the (currently mock)
temporal event detector, and `difficulty_band_points` is a per-band
*placeholder* point value chosen for this draft ruleset. Per the product
requirement ("Do not claim that a model-generated difficulty band is an
official published trick value"), neither the detector's band assignment
nor the ruleset's points-per-band are, or should be presented as, an
official trick-value table from any organization.

## Scoring algorithm (`scoring_engine.py`)

`DeterministicScoringEngine.calculate()` runs four independent, pure
sub-computations, each returning both a numeric result and a list of
human-readable warnings, then blends them:

### 1. Technical points (`_technical_points`)

- Iterates confirmed/predicted events in start-time order.
- Events in `MISTAKE_EVENT_FAMILIES` (e.g. `control_miss`, `landing_miss`,
  `catch_miss`, `yoyo_stop`, `yoyo_change`, `yoyo_detach`) never earn
  positive credit — mistakes are handled entirely through deductions, not
  as negative technical points, keeping the two systems separate and
  auditable.
- Events outside `POSITIVE_EVENT_FAMILIES` (this includes
  `unknown_technical_element`) earn no credit either, but an unclassified
  element specifically adds a review warning so classification quality gets
  human attention.
- Events with `outcome != success` (i.e. `miss` or `uncertain`) earn no
  credit; `uncertain` outcomes add an explicit "excluded pending review"
  warning.
- For every successful, creditable event: `base_points =
  ruleset.difficulty_band_points.points_for(band)`, multiplied by
  `repeated_element_decay.multiplier_for(family, occurrence_index, policy)`
  — see "Repeated-element policies" below for what that multiplier is under
  each policy — and a warning names the exact element, occurrence number,
  and multiplier applied whenever it's below `1.0`. Occurrence counting is
  keyed on `(family, label)`, so distinct tricks in the same family are
  tracked independently.
- `technical_raw` is the unclamped sum; `technical_scaled = min(technical_raw,
  technical_scale_max)`.
- `scoring_engine.technical_points(events, ruleset)` exposes this stage on
  its own (Prompt D requirement 2, "Separate: event detection / technical
  counting / technical scaling / …") for callers — e.g.
  `scoring.pipeline`'s bootstrap confidence-interval resampling — that need
  to re-run just this computation without the rest of `calculate()`.
  `deduction_points(...)` and `freestyle_evaluation_points(...)` are the
  equivalent public entry points for stages 2 and 3 below.

### Repeated-element policies (`RepeatedElementDecay.policy`)

Prompt D requirement 5, "Support repeated-element policies" (plural, not a
single fixed curve). `occurrence_multipliers` (default `[1.0, 0.7, 0.4,
0.2]`, index clamps at the last value for further repeats) is always the
underlying curve; `policy` decides *which* families it applies to and
what happens beyond the curve's length:

| `policy` | Behavior |
| --- | --- |
| `decay_high_risk_only` (default) | Only families in `high_risk_families` (packaged default: `suicide`, `whip_catch`, `horizontal`) decay; every other family earns full credit on every repeat. Matches pre-Prompt-D behavior exactly, so existing rulesets are unaffected unless they opt in. |
| `decay_all_families` | The same curve applies to *every* family, not just the high-risk set — a stricter, research-oriented policy. |
| `full_credit` | Decay disabled outright — appropriate for `practice` mode, where discouraging repetition during practice would be counterproductive. |
| `cap_occurrences` | Like `decay_high_risk_only`, but occurrences beyond `len(occurrence_multipliers)` earn **zero** credit instead of clamping to the last multiplier — a hard repeat cap. |

A `Ruleset`'s own `repeated_element_decay.policy` is the default, but a
caller (in practice, `scoring.profiles.ScoringProfileConfig`) can pass an
explicit `policy` override into `multiplier_for(...)` per computation
without mutating the shared `Ruleset` object — see "Scoring profiles"
below.

### Dangerous-play review flags (`DeductionRule.requires_manual_confirmation`)

Prompt D's major-deduction types are `yoyo_stop`, `yoyo_change`,
`yoyo_detach`, `dangerous_play_review`, and `other`. The packaged ruleset
sets `requires_manual_confirmation: true` only on `dangerous_play_review`
(5.0 pts, uncapped): **"Dangerous-play detection must never automatically
disqualify a player. It must create a review flag."** Concretely,
`scoring_engine.deduction_is_scorable(deduction_type, review_status,
ruleset)` returns `False` for any deduction whose rule sets this flag
unless a human has explicitly set `review_status=CONFIRMED` — a freshly
detected flag (`ReviewStatus.PENDING`, the default) is persisted and
visible for review but contributes exactly `0.0` score impact until then,
by construction rather than convention. `api.services.scoring_service.
recompute_score` filters every deduction through this function before
building the list `DeterministicScoringEngine.calculate` sees; see
`ml/src/yoyovision_ml/scoring/dangerous_play.py` for the heuristic
velocity-based detector that produces these flags in the first place.

### 2. Major deductions (`_deduction_points`)

- Deduction quantities are summed per `DeductionType` first (a single
  routine can have many `MajorDeduction` rows of the same type — "multiple
  deductions per routine" product requirement).
- Each type looks up its `DeductionRule` (points-per-occurrence + optional
  `max_occurrences_penalized`) from the active ruleset; a type with no
  configured rule is skipped with a warning (fails safe, never silently
  guesses a penalty).
- If a cap is configured and exceeded, only the capped quantity is
  penalized (protects against runaway deductions from noisy detection) and
  a warning states the actual vs. penalized count.
- The packaged ruleset's caps: `yoyo_stop` 2.0 pts × up to 6 occurrences,
  `yoyo_change` 3.0 pts × up to 3, `yoyo_detach` 4.0 pts × up to 3,
  `dangerous_play_review` 5.0 pts uncapped (but gated — see "Dangerous-play
  review flags" below), `other` 1.0 pt uncapped.

### 3. Freestyle Evaluation (`_freestyle_evaluation_points`)

- The 8 manual categories (`execution`, `control`, `trick_diversity`,
  `space_use_emphasis`, `music_choreography`, `music_construction`,
  `body_control`, `showmanship`) are entered by a human on a **0–10 scale**
  each; this function multiplies each by its ruleset weight and sums.
- If `freestyle_evaluation` is `None` entirely, the function returns
  `(0.0, 0.0, [warning])` — the placeholder is explicitly `0`, and the
  warning states a human judge must enter values (never silently omitted).
- If some but not all categories are filled in, missing ones are excluded
  from the raw sum (not defaulted to 0) and a warning states how many of 8
  are missing.
- `scaled = (raw / (max_possible_weighted_sum * 10.0)) *
  freestyle_evaluation_scale_max` — i.e. normalized against the maximum
  possible 0–10-per-category score, then rescaled to the ruleset's ceiling.

### 4. Confidence and blending

- `confidence` = the mean of every event's and deduction's `confidence`
  (defaults to `1.0` if there are none, to avoid a misleading `0.0` when
  there's simply nothing to be uncertain about).
- Any event/deduction below `low_confidence_review_threshold` (default
  `0.55`) increments a counter that becomes a single summary warning naming
  the count and threshold.
- `final_score = max(0.0, technical_weight * technical_scaled +
  freestyle_evaluation_weight * fe_scaled - deduction_total)` — deductions
  subtract directly from the blended positive score and the result is
  floored at zero, never negative.

### Audit trail

Every `ScoreBreakdown` returned carries: the exact `ruleset_version` used,
raw and scaled values for both technical and Freestyle Evaluation
components (so the blend is always reconstructible), the total deduction
points, the mean confidence, and the full accumulated `warnings` list
(unofficial-score notice + ruleset disclaimer + every per-computation
warning generated above). Nothing about how the score was derived is
hidden from the reviewing user, and the same events/deductions/evaluation
recomputed under the same ruleset version always yield the same score
(determinism).

## Manual override

Users can add, edit, delete, confirm, or reject any `AnalysisEvent` or
`MajorDeduction` through the API/review UI (`events`/`deductions` routers
and the frontend's event/deduction tables) before triggering
`POST /analyses/{id}/score/recompute`. Because the scoring engine is a pure
function of its inputs, an override is just a change to those inputs
followed by a recompute — there is no separate "override" code path to keep
in sync, and the resulting `ScoreBreakdown` is exactly as auditable as an
unedited one.

## Scoring profiles, overrides, judges, and calibration (`scoring/`)

Prompt D adds `ml/src/yoyovision_ml/scoring/` on top of everything above —
`scoring_engine.py`/`ruleset.py` remain the single source of truth for the
actual point math; this package only changes *which inputs* reach that math
and adds tooling to evaluate it. Kept deliberately `ml`-only for now (no new
API routes, DB tables, or frontend UI — see `docs/adapters.md`'s "Scoring &
judge calibration" section for current scope):

- **Scoring profiles** (`profiles.py`) — `practice` / `judge_assist` /
  `research`, named presets that gate which events are eligible (e.g.
  `research` requires `review_status=CONFIRMED`), which
  `RepeatedElementPolicyType` to use (overriding the ruleset's own
  `policy` without mutating the shared `Ruleset` object), whether automatic
  Freestyle Evaluation estimators run, and whether a bootstrap confidence
  interval is computed. `judge_assist` (the default) matches pre-Prompt-D
  behavior most closely.
- **Per-event manual overrides** (`overrides.py`) — `apply_overrides`
  applies a batch of single-field edits to a copy of the event list (never
  mutating the input), rejecting unknown event IDs or disallowed fields and
  producing a human-readable audit-log entry for every applied or rejected
  edit.
- **Judge clicks and multi-judge scores** (`judges.py`) — aggregates
  several judges' Freestyle Evaluation scores per category (flagging
  disagreement ≥3.0 points), computes pairwise judge-judge agreement, and
  matches manually-entered judge timestamp clicks to detected events within
  a configurable tolerance.
- **Automatic Freestyle Evaluation estimators** (`fe_estimators.py`) —
  optional heuristic estimators for `execution`, `control`,
  `trick_diversity`, `space_use_emphasis`, and `body_control`, each
  reporting a confidence, its supporting features, a model name/version,
  and an explicit "artistic scoring is subjective" warning.
  `music_choreography`/`music_construction` never produce a numeric value
  (no audio analysis exists), and `showmanship` is never auto-estimated at
  all — both stay human-entered by design.
- **Confidence intervals** (`pipeline.py`) — `run_scoring_pipeline`
  orchestrates overrides → profile/confirmation filtering → judge
  aggregation → automatic FE gap-filling → `DeterministicScoringEngine.
  calculate` → an optional bootstrap-resampled `final_score_interval`,
  returning one auditable `ScoringPipelineResult` with every intermediate
  stage's output still visible.
- **Calibration** (`calibration.py`) — mean absolute error, Pearson and
  Spearman correlation, ICC(3,1), Bland–Altman summaries, event-count
  precision/recall, and an optional matplotlib calibration scatter plot for
  comparing model output against expert judges.
- **CLI** (`cli.py`, installed as `yoyovision-scoring`) — `score` runs the
  pipeline over one dataset record; `calibrate` compares a model-prediction
  artifact against a record's adjudicated ground truth and judge scores
  using every metric above. See the README's "Scoring and judge
  calibration" section for example invocations.

## Versioning a new ruleset

1. Add a new YAML file under `ml/src/yoyovision_ml/rulesets/` with a unique
   `version` string.
2. Set `is_official` truthfully and write a real `disclaimer` if you are
   representing a genuinely sanctioned rulebook — otherwise leave both at
   their unofficial-draft defaults.
3. `list_available_rulesets()` will pick it up automatically (it globs the
   directory); no code change is required to make it selectable via
   `get_ruleset_by_version()` or the `/rulesets` API.
4. Existing `ScoreBreakdown` rows are unaffected — they retain the
   `ruleset_version` they were computed under. Recomputing a score under a
   different ruleset version is an explicit user action, never implicit.
