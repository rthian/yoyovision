"""Always-available (no `torch`) `TemporalEventDetector` baselines, per
Prompt C's required baseline comparisons: "majority class" and "hand-crafted
threshold rules".

Neither of these is a trained model -- `MajorityClassEventDetector` counts
label frequency in whatever `TrainingSample`s it is `.fit()` on;
`ThresholdRuleEventDetector` never looks at any training data at all, only
fixed velocity/direction/hand-distance thresholds on Prompt B's kinematic
features. Both exist purely as a weak reference point the real TCN model
must be compared against (Prompt C: "Do not claim production readiness based
only on clip-level accuracy") -- neither should ever be presented as a
usable production detector.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np

from yoyovision_ml.adapters_registry import register_temporal_event_detector
from yoyovision_ml.domain import (
    AnalysisEventPrediction,
    DeductionPrediction,
    EventFamily,
    FeatureSet,
    Outcome,
)
from yoyovision_ml.events.config import InferenceConfig
from yoyovision_ml.events.convert import to_analysis_event_prediction
from yoyovision_ml.events.decode import decode_predictions
from yoyovision_ml.events.labels import CLASS_TO_INDEX, NUM_CLASSES, NUM_OUTCOMES, OUTCOME_CLASSES
from yoyovision_ml.events.types import EventDetection, TrainingSample
from yoyovision_ml.events.windowing import feature_matrix, frame_timestamps_ms
from yoyovision_ml.perception.features import (
    FEATURE_HAND_DISTANCE,
    FEATURE_YOYO_DIRECTION_DEG,
    FEATURE_YOYO_VELOCITY,
)


@register_temporal_event_detector("majority")
class MajorityClassEventDetector:
    """Predicts exactly one event spanning the entire clip: the single most
    frequent (class, outcome) pair seen at `.fit()` time, or
    `unknown_technical_element`/`uncertain` if never fit. The simplest
    possible temporal-detection baseline."""

    model_name = "majority-class-baseline"
    model_version = "1.0.0-not-a-trained-model"

    def __init__(
        self,
        majority_family: EventFamily = EventFamily.UNKNOWN_TECHNICAL_ELEMENT,
        majority_outcome: Outcome = Outcome.UNCERTAIN,
        confidence: float = 0.5,
    ) -> None:
        self._family = majority_family
        self._outcome = majority_outcome
        self._confidence = confidence

    @classmethod
    def fit(cls, samples: Iterable[TrainingSample]) -> MajorityClassEventDetector:
        family_counts: Counter[EventFamily] = Counter()
        outcome_counts: Counter[Outcome] = Counter()
        total = 0
        for sample in samples:
            for event in sample.trick_events:
                if event.family not in CLASS_TO_INDEX:
                    continue
                family_counts[event.family] += 1
                outcome_counts[event.outcome] += 1
                total += 1
        if total == 0:
            return cls()
        family, count = family_counts.most_common(1)[0]
        outcome = outcome_counts.most_common(1)[0][0]
        return cls(majority_family=family, majority_outcome=outcome, confidence=count / total)

    def predict(
        self, features: FeatureSet
    ) -> tuple[list[AnalysisEventPrediction], list[DeductionPrediction]]:
        if not features.frames:
            return [], []
        start_ms = features.frames[0].frame_ms
        end_ms = features.frames[-1].frame_ms
        if end_ms <= start_ms:
            return [], []
        detection = EventDetection(
            label=self._family.value,
            family=self._family,
            start_ms=start_ms,
            end_ms=end_ms,
            outcome=self._outcome,
            confidence=round(self._confidence, 4),
            model_version=self.model_version,
            supporting_frame_range=(start_ms, end_ms),
        )
        return [to_analysis_event_prediction(detection, self.model_name)], []


@register_temporal_event_detector("rules")
class ThresholdRuleEventDetector:
    """Hand-crafted, fixed-threshold heuristic over `yoyo_velocity`,
    `yoyo_direction_deg`, and `hand_distance`. Never trained, never
    calibrated -- rule firings are treated as probability `1.0`, so this
    detector's "confidence" is meaningless as a calibration signal and exists
    only so it can flow through the shared `decode.decode_predictions` path.
    Every predicted outcome is `uncertain`: a fixed threshold rule has no
    basis for judging success/miss."""

    model_name = "threshold-rule-baseline"
    model_version = "1.0.0-not-a-trained-model"

    def __init__(
        self,
        config: InferenceConfig | None = None,
        velocity_high: float = 0.5,
        velocity_very_high: float = 0.8,
        velocity_low: float = 0.05,
        hand_distance_high: float = 0.3,
        vertical_direction_tolerance_deg: float = 30.0,
    ) -> None:
        self._config = config or InferenceConfig()
        self._velocity_high = velocity_high
        self._velocity_very_high = velocity_very_high
        self._velocity_low = velocity_low
        self._hand_distance_high = hand_distance_high
        self._vertical_tolerance = vertical_direction_tolerance_deg

    def predict(
        self, features: FeatureSet
    ) -> tuple[list[AnalysisEventPrediction], list[DeductionPrediction]]:
        if not features.frames:
            return [], []

        frame_ms = frame_timestamps_ms(features)
        velocity = feature_matrix(features, (FEATURE_YOYO_VELOCITY,))[:, 0]
        direction = feature_matrix(features, (FEATURE_YOYO_DIRECTION_DEG,))[:, 0]
        hand_distance = feature_matrix(features, (FEATURE_HAND_DISTANCE,))[:, 0]

        class_probs = np.zeros((len(frame_ms), NUM_CLASSES), dtype=np.float64)

        vertical = np.abs(np.abs(direction) - 90.0) < self._vertical_tolerance
        hop_mask = (velocity > self._velocity_high) & vertical
        class_probs[hop_mask, CLASS_TO_INDEX[EventFamily.HOP]] = 1.0

        whip_mask = (velocity > self._velocity_very_high) & ~hop_mask
        class_probs[whip_mask, CLASS_TO_INDEX[EventFamily.WHIP_CATCH]] = 1.0

        slack_mask = (velocity < self._velocity_low) & (hand_distance > self._hand_distance_high)
        class_probs[slack_mask, CLASS_TO_INDEX[EventFamily.SLACK]] = 1.0

        outcome_probs = np.zeros((len(frame_ms), NUM_OUTCOMES), dtype=np.float64)
        outcome_probs[:, OUTCOME_CLASSES.index("uncertain")] = 1.0

        detections = decode_predictions(
            frame_ms, class_probs, outcome_probs, self.model_version, self._config
        )
        events = [to_analysis_event_prediction(d, self.model_name) for d in detections]
        return events, []
