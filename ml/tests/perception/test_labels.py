from __future__ import annotations

from yoyovision_ml.domain import BoundingBox
from yoyovision_ml.perception.labels import target_bbox_from_annotation


def test_target_bbox_from_point_uses_fixed_box_size() -> None:
    bbox, visible = target_bbox_from_annotation(
        point_x=0.5,
        point_y=0.5,
        bbox=None,
        visibility="visible",
        point_box_size=0.1,
    )
    assert visible is True
    assert bbox == (0.45, 0.45, 0.1, 0.1)


def test_target_bbox_from_annotation_marks_occluded_invisible() -> None:
    bbox, visible = target_bbox_from_annotation(
        point_x=0.5,
        point_y=0.5,
        bbox=BoundingBox(x=0.4, y=0.4, width=0.1, height=0.1),
        visibility="fully_occluded",
        point_box_size=0.1,
    )
    assert visible is False
    assert bbox == (0.0, 0.0, 0.0, 0.0)
