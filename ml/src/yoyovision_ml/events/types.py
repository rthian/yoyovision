"""Plain result/input dataclasses for the events package.

Kept separate from `yoyovision_ml.domain` (which stays framework-agnostic and
shared by `api`/`workers`) because these types are specific to *how* this
package's models are trained and decode raw predictions -- `EventDetection`
carries extra fields (`needs_review`, `supporting_frame_range`) that the
persisted `domain.AnalysisEventPrediction` deliberately does not, and
`TrainingSample` bundles training-only bookkeeping (`player_id`) that has no
place in a runtime prediction type.
"""

from __future__ import annotations

from dataclasses import dataclass

from yoyovision_ml.dataset.schema import TrickEventAnnotation
from yoyovision_ml.domain import EventFamily, FeatureSet, Outcome


@dataclass(slots=True, frozen=True)
class EventDetection:
    """One decoded event, matching Prompt C's "INFERENCE" return contract
    (`label`, `start_ms`, `end_ms`, `outcome`, `confidence`, `model_version`,
    `supporting_frame_range`) plus `needs_review` for the uncertainty-routing
    requirement."""

    label: str
    family: EventFamily
    start_ms: int
    end_ms: int
    outcome: Outcome
    confidence: float
    model_version: str
    #: `(first_frame_ms, last_frame_ms)` of the input frames this detection
    #: was decoded from -- the "supporting frame range" evidence pointer.
    supporting_frame_range: tuple[int, int]
    #: Set when `confidence` fell below `InferenceConfig.uncertainty_threshold`
    #: and `uncertainty_action == "flag_review"` (relabelling to
    #: `unknown_technical_element` is the other configured action, which
    #: instead changes `label`/`family` directly rather than setting this).
    needs_review: bool = False


@dataclass(slots=True, frozen=True)
class TrainingSample:
    """One labelled training clip: Prompt-B perception features plus
    Prompt-A ground-truth trick events, grouped by `player_id` so
    `train.player_grouped_split` can keep a player's clips in exactly one
    split (Prompt C: "player-grouped data splits", "no train/test leakage")."""

    video_id: str
    player_id: str
    features: FeatureSet
    trick_events: tuple[TrickEventAnnotation, ...]
