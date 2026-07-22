from __future__ import annotations

from pathlib import Path

import pytest

from yoyovision_ml.dataset.importers.cvat import CvatImportError, import_cvat_yoyo_track
from yoyovision_ml.dataset.schema import VisibilityState

SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <track id="0" label="yoyo" source="manual">
    <box frame="0" outside="0" occluded="0" keyframe="1"
         xtl="960.0" ytl="540.0" xbr="1020.0" ybr="600.0"></box>
    <box frame="1" outside="0" occluded="1" keyframe="1"
         xtl="970.0" ytl="545.0" xbr="1030.0" ybr="605.0"></box>
    <box frame="2" outside="1" occluded="0" keyframe="1"
         xtl="0.0" ytl="0.0" xbr="0.0" ybr="0.0"></box>
  </track>
  <track id="1" label="left_hand" source="manual">
    <points frame="0" outside="0" occluded="0" keyframe="1" points="400.0,300.0"></points>
  </track>
</annotations>
"""


@pytest.fixture
def sample_xml_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.xml"
    path.write_text(SAMPLE_XML, encoding="utf-8")
    return path


def test_import_cvat_yoyo_track_parses_all_frames(sample_xml_path: Path) -> None:
    annotations = import_cvat_yoyo_track(
        sample_xml_path, fps=30.0, frame_width=1920, frame_height=1080
    )
    assert len(annotations) == 3
    assert [a.frame_ms for a in annotations] == [0, 33, 67]


def test_import_cvat_yoyo_track_maps_visibility_states(sample_xml_path: Path) -> None:
    annotations = import_cvat_yoyo_track(
        sample_xml_path, fps=30.0, frame_width=1920, frame_height=1080
    )
    assert annotations[0].visibility == VisibilityState.VISIBLE
    assert annotations[1].visibility == VisibilityState.PARTIALLY_OCCLUDED
    assert annotations[2].visibility == VisibilityState.OUTSIDE_FRAME
    assert annotations[2].bbox is None
    assert annotations[2].point is None


def test_import_cvat_yoyo_track_normalizes_bbox_coordinates(sample_xml_path: Path) -> None:
    annotations = import_cvat_yoyo_track(
        sample_xml_path, fps=30.0, frame_width=1920, frame_height=1080
    )
    bbox = annotations[0].bbox
    assert bbox is not None
    assert bbox.x == pytest.approx(960.0 / 1920)
    assert bbox.y == pytest.approx(540.0 / 1080)
    assert bbox.width == pytest.approx(60.0 / 1920)
    assert bbox.height == pytest.approx(60.0 / 1080)


def test_import_cvat_yoyo_track_raises_for_unknown_label(sample_xml_path: Path) -> None:
    with pytest.raises(CvatImportError):
        import_cvat_yoyo_track(
            sample_xml_path,
            fps=30.0,
            frame_width=1920,
            frame_height=1080,
            track_label="does_not_exist",
        )


def test_import_cvat_points_track(tmp_path: Path) -> None:
    path = tmp_path / "points.xml"
    path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <track id="0" label="yoyo" source="manual">
    <points frame="0" outside="0" occluded="0" keyframe="1" points="500.0,400.0"></points>
  </track>
</annotations>
""",
        encoding="utf-8",
    )
    annotations = import_cvat_yoyo_track(path, fps=25.0, frame_width=1000, frame_height=800)
    assert len(annotations) == 1
    assert annotations[0].point is not None
    assert annotations[0].point.x == pytest.approx(0.5)
    assert annotations[0].point.y == pytest.approx(0.5)
