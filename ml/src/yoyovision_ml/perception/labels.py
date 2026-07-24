"""Label helpers for yo-yo detector training."""

from __future__ import annotations

from yoyovision_ml.domain import BoundingBox

_INVISIBLE_VISIBILITIES = frozenset({"fully_occluded", "outside_frame", "unlabelled"})


def is_visible_visibility(visibility: object) -> bool:
    return str(visibility) not in _INVISIBLE_VISIBILITIES


def target_bbox_from_annotation(
    *,
    point_x: float | None,
    point_y: float | None,
    bbox: BoundingBox | None,
    visibility: object,
    point_box_size: float,
) -> tuple[tuple[float, float, float, float], bool]:
    visible = is_visible_visibility(visibility)
    if not visible:
        return (0.0, 0.0, 0.0, 0.0), False
    if bbox is not None:
        return (bbox.x, bbox.y, bbox.width, bbox.height), True
    if point_x is not None and point_y is not None:
        half = point_box_size / 2.0
        return (point_x - half, point_y - half, point_box_size, point_box_size), True
    return (0.0, 0.0, 0.0, 0.0), False
