"""Canonical class vocabulary and feature-column subsets for Prompt C.

`EVENT_CLASSES` is deliberately a *subset* of `domain.EventFamily` -- the
temporal trick-event model predicts only the 20 atomic 1A trick-element
families named in Prompt C. The 3 equipment families (`yoyo_stop`,
`yoyo_change`, `yoyo_detach`) are excluded: those are already handled as
`DeductionPrediction`s elsewhere (see `adapters_mock.MockTemporalEventDetector`
and `docs/ruleset.md`), and mixing a binary "did the yo-yo detach" signal into
a 20-way multi-label trick classifier would conflate two different label
spaces. A trained checkpoint from this package always returns an empty
deduction list; equipment-event detection is a separate, future model.
"""

from __future__ import annotations

from yoyovision_ml.domain import EQUIPMENT_EVENT_FAMILIES, EventFamily
from yoyovision_ml.multimodal.features import MULTIMODAL_FEATURE_NAMES
from yoyovision_ml.perception.features import ALL_FEATURE_NAMES

#: The 20 Prompt C classes, in a fixed, stable order (index == model output
#: channel). Order matches the prompt's "INITIAL CLASSES" list exactly so
#: checkpoint channel indices are human-traceable without a lookup table.
EVENT_CLASSES: tuple[EventFamily, ...] = tuple(
    family for family in EventFamily if family not in EQUIPMENT_EVENT_FAMILIES
)

assert len(EVENT_CLASSES) == 20, f"expected 20 Prompt C classes, got {len(EVENT_CLASSES)}"

#: EventFamily <-> model channel index, both directions.
CLASS_TO_INDEX: dict[EventFamily, int] = {family: i for i, family in enumerate(EVENT_CLASSES)}
INDEX_TO_CLASS: dict[int, EventFamily] = {i: family for family, i in CLASS_TO_INDEX.items()}

NUM_CLASSES = len(EVENT_CLASSES)

#: Outcome classes the outcome head predicts, in fixed order.
OUTCOME_CLASSES: tuple[str, ...] = ("success", "miss", "uncertain")
NUM_OUTCOMES = len(OUTCOME_CLASSES)

#: Feature-column subsets for the "skeleton-only" / "yo-yo-trajectory-only" /
#: "fused" ablation baselines Prompt C asks for. Column names are Prompt B's
#: `perception.features.ALL_FEATURE_NAMES` -- every column whose name starts
#: with `yoyo_` describes the yo-yo's own trajectory/track quality; everything
#: else (hand distance, wrist velocity, elbow angles, shoulder width, stage
#: coordinates) describes body/skeleton geometry. `stage_x`/`stage_y` are
#: yoyo-position-relative-to-body, i.e. they need *both* signals -- kept in
#: both subsets since excluding them from either ablation would silently
#: change what "skeleton-only" or "trajectory-only" means from run to run.
TRAJECTORY_FEATURES: tuple[str, ...] = tuple(
    name for name in ALL_FEATURE_NAMES if name.startswith("yoyo_")
) + ("stage_x", "stage_y")
SKELETON_FEATURES: tuple[str, ...] = tuple(
    name for name in ALL_FEATURE_NAMES if not name.startswith("yoyo_")
) + ("stage_x", "stage_y")
FUSED_FEATURES: tuple[str, ...] = ALL_FEATURE_NAMES

#: Prompt E aliases/additions. `"fused"` above is Prompt C's terminology
#: for "every *kinematic* column" (skeleton + trajectory) -- to avoid
#: confusing that with Prompt E's "kinematics vs. multimodal" ablation axis,
#: `KINEMATICS_ONLY_FEATURES` is an explicitly-named alias of the same
#: columns, and `MULTIMODAL_FUSED_FEATURES` adds Prompt E's RGB/string-
#: segmentation/audio columns (`multimodal.features.MULTIMODAL_FEATURE_NAMES`)
#: on top.
KINEMATICS_ONLY_FEATURES: tuple[str, ...] = ALL_FEATURE_NAMES
MULTIMODAL_FUSED_FEATURES: tuple[str, ...] = ALL_FEATURE_NAMES + MULTIMODAL_FEATURE_NAMES

FEATURE_SUBSETS: dict[str, tuple[str, ...]] = {
    "fused": FUSED_FEATURES,
    "skeleton": SKELETON_FEATURES,
    "trajectory": TRAJECTORY_FEATURES,
    "kinematics_only": KINEMATICS_ONLY_FEATURES,
    "multimodal_fused": MULTIMODAL_FUSED_FEATURES,
}
