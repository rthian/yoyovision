from __future__ import annotations

from yoyovision_ml.domain import EQUIPMENT_EVENT_FAMILIES, EventFamily
from yoyovision_ml.events.labels import (
    CLASS_TO_INDEX,
    EVENT_CLASSES,
    FEATURE_SUBSETS,
    FUSED_FEATURES,
    INDEX_TO_CLASS,
    KINEMATICS_ONLY_FEATURES,
    MULTIMODAL_FUSED_FEATURES,
    NUM_CLASSES,
    NUM_OUTCOMES,
    OUTCOME_CLASSES,
    SKELETON_FEATURES,
    TRAJECTORY_FEATURES,
)
from yoyovision_ml.multimodal.features import MULTIMODAL_FEATURE_NAMES
from yoyovision_ml.perception.features import ALL_FEATURE_NAMES

_PROMPT_C_CLASS_ORDER = (
    "mount",
    "hop",
    "laceration",
    "whip_catch",
    "slack",
    "suicide",
    "rejection",
    "roll",
    "underpass",
    "overpass",
    "bind",
    "return",
    "regeneration",
    "horizontal",
    "fingerspin",
    "body_trick",
    "control_miss",
    "landing_miss",
    "catch_miss",
    "unknown_technical_element",
)


def test_event_classes_has_exactly_twenty_entries() -> None:
    assert len(EVENT_CLASSES) == 20
    assert NUM_CLASSES == 20


def test_event_classes_matches_prompt_c_order_exactly() -> None:
    assert tuple(family.value for family in EVENT_CLASSES) == _PROMPT_C_CLASS_ORDER


def test_event_classes_excludes_equipment_families() -> None:
    for family in EQUIPMENT_EVENT_FAMILIES:
        assert family not in EVENT_CLASSES


def test_class_to_index_and_index_to_class_are_inverses() -> None:
    assert len(CLASS_TO_INDEX) == NUM_CLASSES
    for family, index in CLASS_TO_INDEX.items():
        assert INDEX_TO_CLASS[index] is family
    for index, family in INDEX_TO_CLASS.items():
        assert CLASS_TO_INDEX[family] == index


def test_class_to_index_covers_indices_zero_to_num_classes_minus_one() -> None:
    assert sorted(CLASS_TO_INDEX.values()) == list(range(NUM_CLASSES))


def test_outcome_classes_are_success_miss_uncertain() -> None:
    assert OUTCOME_CLASSES == ("success", "miss", "uncertain")
    assert NUM_OUTCOMES == 3


def test_fused_features_is_every_perception_feature_column() -> None:
    assert FUSED_FEATURES == ALL_FEATURE_NAMES


def test_trajectory_features_only_contains_yoyo_or_stage_columns() -> None:
    for name in TRAJECTORY_FEATURES:
        assert name.startswith("yoyo_") or name in ("stage_x", "stage_y")


def test_skeleton_features_excludes_yoyo_only_columns() -> None:
    for name in SKELETON_FEATURES:
        assert not name.startswith("yoyo_") or name in ("stage_x", "stage_y")


def test_skeleton_and_trajectory_features_together_cover_every_fused_column() -> None:
    covered = set(SKELETON_FEATURES) | set(TRAJECTORY_FEATURES)
    assert covered == set(FUSED_FEATURES)


def test_feature_subsets_dict_has_fused_skeleton_trajectory_keys() -> None:
    assert set(FEATURE_SUBSETS) == {
        "fused",
        "skeleton",
        "trajectory",
        "kinematics_only",
        "multimodal_fused",
    }
    assert FEATURE_SUBSETS["fused"] == FUSED_FEATURES
    assert FEATURE_SUBSETS["skeleton"] == SKELETON_FEATURES
    assert FEATURE_SUBSETS["trajectory"] == TRAJECTORY_FEATURES
    assert FEATURE_SUBSETS["kinematics_only"] == KINEMATICS_ONLY_FEATURES
    assert FEATURE_SUBSETS["multimodal_fused"] == MULTIMODAL_FUSED_FEATURES


def test_kinematics_only_features_is_an_alias_of_fused_features() -> None:
    assert KINEMATICS_ONLY_FEATURES == FUSED_FEATURES == ALL_FEATURE_NAMES


def test_multimodal_fused_features_extends_kinematics_with_multimodal_columns() -> None:
    assert MULTIMODAL_FUSED_FEATURES == ALL_FEATURE_NAMES + MULTIMODAL_FEATURE_NAMES
    assert len(MULTIMODAL_FUSED_FEATURES) == len(ALL_FEATURE_NAMES) + len(
        MULTIMODAL_FEATURE_NAMES
    )
    # No accidental name collisions between kinematic and multimodal columns.
    assert set(ALL_FEATURE_NAMES) & set(MULTIMODAL_FEATURE_NAMES) == set()


def test_every_event_class_is_a_valid_event_family() -> None:
    for family in EVENT_CLASSES:
        assert isinstance(family, EventFamily)
