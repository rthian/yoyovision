# Annotation Handbook (v1)

This is the annotator-facing guide for labelling 1A freestyle footage into
the YoYoVision dataset format (`ml/src/yoyovision_ml/dataset/schema.py`).
It assumes you know 1A yo-yo play but not the codebase -- read this before
annotating your first video, and re-read the ontology section whenever a
new ontology version ships.

## ⚠️ What this dataset is and isn't

- **Not an official trick registry.** The label set in
  `ml/src/yoyovision_ml/dataset/ontology/v1.yaml` uses common community
  trick names as *examples per family*; it is not sourced from or endorsed
  by IYYF, WYYC, or any competition body (see `docs/ruleset.md` for the
  same disclaimer applied to scoring).
- **Ground truth, not a suggestion.** Whatever you annotate becomes
  training/evaluation data for future models. If you're unsure, mark the
  event `unknown_technical_element` / outcome `uncertain` rather than
  guessing a specific label -- a wrong confident label is worse than an
  honest "unknown."

## Before you start

1. Confirm which **ontology version** you're annotating against (currently
   `dataset-ontology-v1`) and which **dataset directory** you're adding to.
2. Get an `annotator_id` (a stable handle, e.g. your initials + a number if
   there's a name collision) -- every annotation you make is attributed to
   it via `AnnotationProvenance.annotator_id`.
3. Know the video's `player_id`. **Never invent a new `player_id` for a
   performer who already has one in the dataset** -- this is what makes
   player-grouped train/val/test splits (`dataset/splits.py`) actually
   prevent a model from memorizing a specific person.

## What you are annotating

For each video, you produce one `DatasetRecord` (a JSON file). A record
holds, all keyed by integer **milliseconds** from the start of the video:

| Stream | What it captures |
|---|---|
| `trick_events` | Time-bounded atomic elements: mount, hop, laceration, whip_catch, slack, suicide, rejection, roll, underpass, overpass, bind, return, regeneration, horizontal, fingerspin, body_trick -- plus mistake/equipment events: control_miss, landing_miss, catch_miss, yoyo_stop, yoyo_change, yoyo_detach, unknown_technical_element |
| `deductions` | Discrete scoring-relevant moments (equipment failures, form issues) |
| `yoyo_track` | Per-frame yo-yo position (point or box) + visibility |
| `pose_landmarks` / `hand_landmarks` | Body/hand keypoints, only where you're specifically asked to annotate them (most annotators will only ever touch `trick_events`/`yoyo_track`) |
| `string_masks` | Frame-level string visibility/segmentation, only for string-focused annotation passes |
| `judge_clicks` | A judge's raw real-time perceived-event timestamp, no boundaries |
| `freestyle_evaluations` | A judge's 0-10 scorecard per Freestyle Evaluation category |

You will typically be assigned ONE of these streams to focus on for a given
pass, not all of them at once.

## Annotating a trick event

1. **Start (`start_ms`) and end (`end_ms`)**: start at the first frame where
   the element is unambiguously beginning (e.g. the yo-yo leaving the
   string plane for a hop); end at the first frame where it's clearly
   resolved (landed/caught/failed). `end_ms` must be strictly greater than
   `start_ms` -- a zero-length event will be rejected.
2. **`label`**: pick the most specific ontology label that fits. If nothing
   fits, use `unclassified_element` (family `unknown_technical_element`) and
   add a note describing what you saw -- do NOT force it into the nearest
   wrong label.
3. **`family`**: must match the ontology's family for your chosen label
   (the dataset validator checks this automatically -- if it disagrees with
   you, you likely picked the wrong label, not a wrong family).
4. **`outcome`**: `success`, `miss`, or `uncertain`. Use `uncertain` for
   genuinely ambiguous footage (e.g. camera angle makes a catch unclear) --
   never coerce ambiguity into `success` or `miss` to avoid inconsistent
   ground truth.
5. **`difficulty_band`**: `basic` / `intermediate` / `advanced` if you're
   confident, else `unknown`. This is an internal training signal, never an
   official difficulty value (see `docs/ruleset.md`).
6. **Overlap rule**: two events of the **same family** for the same
   performer should not overlap in time unless the ontology explicitly
   allows it for that family (currently `body_trick` and `horizontal`,
   because those often legitimately contain a nested element). If the
   validator flags an overlap you believe is legitimate, that's a signal to
   propose an ontology change, not to work around the validator.

## Visibility states

Every `yoyo_track` frame and every landmark keypoint carries one of:

- `visible` -- clearly visible, annotate its position normally.
- `partially_occluded` -- visible but obstructed (e.g. behind a finger);
  still annotate a best-effort position.
- `fully_occluded` -- not visible but known to be in-frame (e.g. behind the
  body); position may be omitted.
- `outside_frame` -- has left the camera's field of view; position may be
  omitted.
- `unlabelled` -- reserved for frames nobody has annotated yet; do not use
  this to mean "occluded," use one of the occlusion states instead.

**Never guess a position for `fully_occluded`/`outside_frame` frames.** The
schema allows omitting `point`/`bbox` specifically so occlusion is
represented honestly rather than papered over with an interpolated guess --
interpolation, if used at all, happens later in the perception pipeline
(Prompt B), not in ground-truth annotation.

## String masks

If you are doing a string-focused pass: `observable=True` requires a real
`mask_key` (a reference to a saved mask asset, never inline pixel data);
`observable=False` requires `mask_key=None`. **Never mark a frame
`observable=True` with a guessed/hallucinated mask** -- if the string isn't
reliably traceable in that frame, mark it `observable=False` and move on.

## Multiple annotators and adjudication

Some videos are annotated independently by two annotators before a lead
reviewer adjudicates. If you're doing an independent pass:

- Save your own `DatasetRecord` with `is_adjudicated=False` and your own
  `annotator_id` -- do not look at another annotator's file first.
- Leave adjudication (`is_adjudicated=True`, `adjudicated_by=<your handle>`,
  and a filled-in `adjudication_notes` explaining what was merged/changed)
  to whoever is designated as the reviewer for that video.

`dataset stats` (the CLI's `stats` command) reports a per-video-pair
agreement ratio for un-adjudicated passes -- a low ratio is a signal that
video needs review attention, not that either annotator "failed."

## Judge clicks and Freestyle Evaluation

- `judge_clicks` are intentionally low-effort: just the judge's real-time
  perceived timestamp and, optionally, what they thought they saw. Do not
  try to retroactively make a click "line up" with a trick_event boundary
  -- the whole point is comparing raw judge perception against detected
  boundaries later (see the ML roadmap's judge-calibration phase).
- `freestyle_evaluations` categories (`execution`, `control`,
  `trick_diversity`, `space_use_emphasis`, `music_choreography`,
  `music_construction`, `body_control`, `showmanship`) are entered on a
  **0-10 scale**, matching `scoring_engine.py`'s existing assumption. Leave
  a category `None` (not `0`) if the judge didn't evaluate it -- `0` means
  "evaluated and scored zero," which is a different fact.

## Tools

```bash
cd ml
# Validate a whole dataset directory (errors block use; warnings don't):
python -m yoyovision_ml.dataset.cli validate <dataset_dir>

# See class distribution, split breakdown, and annotator agreement:
python -m yoyovision_ml.dataset.cli stats <dataset_dir>

# Generate (and optionally --write) player-grouped train/val/test splits:
python -m yoyovision_ml.dataset.cli split <dataset_dir> --seed 42

# Import a CVAT video-XML bounding-box/point track as a yo-yo track
# (boxes/points only -- does not import CVAT skeleton exports):
python -m yoyovision_ml.dataset.cli import-cvat <xml_path> \
    --fps 30 --width 1920 --height 1080 --output out.json
```

Run `validate` after every annotation session, before handing a video off
for adjudication or merging it into a shared dataset directory. A dataset
with `ERROR`-severity issues should be treated as not-yet-usable; `WARNING`s
(currently: on-disk video checksum drift) are informational.

## See also

- `ml/sample_data/dataset_v1/README.md` -- a fully worked, synthetic example
  dataset exercising every field described above (placeholder video files,
  hand-authored annotation values -- not real footage).
- `ml/src/yoyovision_ml/dataset/ontology/v1.yaml` -- the current label set.
- `docs/data_model.md` -- how the *runtime* pipeline's vocabulary
  (`EventFamily`, `Outcome`, `DifficultyBand`, etc.) relates to this
  dataset schema (they're intentionally the same enums).
