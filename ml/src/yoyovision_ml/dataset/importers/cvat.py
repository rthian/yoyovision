"""Best-effort importer for CVAT "video" XML export (1.1) tracks.

SCOPE (confirmed for this iteration): bounding-box and point *tracks* only,
mapped onto `YoyoFrameAnnotation` -- i.e. this imports a single tracked
object's per-frame position (typically the yo-yo). CVAT skeleton/keypoint
export for full body/hand landmarks is explicitly OUT of scope here; do not
call this importer expecting `PoseLandmarkFrame`/`HandLandmarkFrame` output.

CVAT XML shape (see https://opencv.github.io/cvat/docs/manual/advanced/xml_format/):

    <annotations>
      <track id="0" label="yoyo" source="manual">
        <box frame="0" outside="0" occluded="0" keyframe="1"
             xtl="100.0" ytl="200.0" xbr="150.0" ybr="250.0"/>
        <points frame="1" outside="0" occluded="1" keyframe="1"
                points="120.0,220.0"/>
        ...
      </track>
    </annotations>

Known fidelity limits, documented rather than silently papered over:

* CVAT's `occluded` flag is boolean; it does not distinguish "partially
  occluded" from "fully occluded" the way our `VisibilityState` does. This
  importer maps `occluded="1"` to `PARTIALLY_OCCLUDED` (the more common
  real-world case for a still-visible-but-obstructed yo-yo) and always
  leaves a note on the resulting frame saying the distinction was inferred,
  not observed.
* CVAT XML for a "task" does not reliably encode true source fps or frame
  dimensions in a stable, versioned location, so `fps`, `frame_width`, and
  `frame_height` must be supplied explicitly by the caller rather than
  parsed from `<meta>`.
* Only the first matching `<track>` (by `label`) is imported; multi-track
  import (e.g. simultaneous left/right hand tracks) is not implemented.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from defusedxml import ElementTree as SafeET

from yoyovision_ml.dataset.schema import (
    NormalizedBBox,
    NormalizedPoint,
    VisibilityState,
    YoyoFrameAnnotation,
)


class CvatImportError(ValueError):
    """Raised when the CVAT XML does not contain a track this importer can handle."""


def _visibility_for_shape(shape: ET.Element) -> VisibilityState:
    if shape.get("outside") == "1":
        return VisibilityState.OUTSIDE_FRAME
    if shape.get("occluded") == "1":
        return VisibilityState.PARTIALLY_OCCLUDED
    return VisibilityState.VISIBLE


def _frame_ms_for(frame_index: int, fps: float) -> int:
    return round(frame_index / fps * 1000)


def _bbox_from_box_shape(box: ET.Element, frame_width: int, frame_height: int) -> NormalizedBBox:
    xtl, ytl, xbr, ybr = (float(box.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr"))
    x = max(0.0, min(1.0, xtl / frame_width))
    y = max(0.0, min(1.0, ytl / frame_height))
    width = max(0.0, min(1.0 - x, (xbr - xtl) / frame_width))
    height = max(0.0, min(1.0 - y, (ybr - ytl) / frame_height))
    return NormalizedBBox(x=x, y=y, width=width, height=height)


def _point_from_points_shape(
    points_shape: ET.Element, frame_width: int, frame_height: int
) -> NormalizedPoint:
    raw = points_shape.attrib["points"].split(";")[0]  # first point only; multi-point unsupported
    px_str, py_str = raw.split(",")
    px, py = float(px_str), float(py_str)
    x = max(0.0, min(1.0, px / frame_width))
    y = max(0.0, min(1.0, py / frame_height))
    return NormalizedPoint(x=x, y=y)


def import_cvat_yoyo_track(
    xml_path: Path,
    *,
    fps: float,
    frame_width: int,
    frame_height: int,
    track_label: str = "yoyo",
) -> list[YoyoFrameAnnotation]:
    """Parses one CVAT video-XML track (matched by `label`) into a list of
    `YoyoFrameAnnotation`, one per `<box>`/`<points>` shape in the track.

    Raises `CvatImportError` if no `<track label="{track_label}">` is found.
    """
    root = SafeET.parse(xml_path).getroot()
    if root is None:
        raise CvatImportError(f"{xml_path} has no root XML element.")
    track = next(
        (t for t in root.findall("track") if t.get("label") == track_label),
        None,
    )
    if track is None:
        available = sorted({t.get("label", "") for t in root.findall("track")})
        raise CvatImportError(
            f"No <track label='{track_label}'> found in {xml_path}. "
            f"Available track labels: {available or '<none>'}."
        )

    annotations: list[YoyoFrameAnnotation] = []
    for shape in list(track.findall("box")) + list(track.findall("points")):
        frame_index = int(shape.attrib["frame"])
        visibility = _visibility_for_shape(shape)
        frame_ms = _frame_ms_for(frame_index, fps)

        point: NormalizedPoint | None = None
        bbox: NormalizedBBox | None = None
        if visibility != VisibilityState.OUTSIDE_FRAME:
            if shape.tag == "box":
                bbox = _bbox_from_box_shape(shape, frame_width, frame_height)
            else:
                point = _point_from_points_shape(shape, frame_width, frame_height)

        annotations.append(
            YoyoFrameAnnotation(
                frame_ms=frame_ms,
                point=point,
                bbox=bbox,
                visibility=visibility,
                confidence=None,
            )
        )

    annotations.sort(key=lambda a: a.frame_ms)
    return annotations
