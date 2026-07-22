from __future__ import annotations

import json
from pathlib import Path

from yoyovision_ml.dataset.cli import main

SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <track id="0" label="yoyo" source="manual">
    <box frame="0" outside="0" occluded="0" keyframe="1"
         xtl="960.0" ytl="540.0" xbr="1020.0" ybr="600.0"></box>
  </track>
</annotations>
"""


def test_merge_cvat_writes_yoyo_track_into_record(tmp_path: Path) -> None:
    dataset_dir = Path("ml/sample_data/dataset_v1")
    xml_path = tmp_path / "track.xml"
    xml_path.write_text(SAMPLE_XML, encoding="utf-8")
    out_dir = tmp_path / "dataset"
    out_dir.mkdir()
    (out_dir / "records").mkdir()
    record = json.loads(
        (dataset_dir / "records/sample_video_002__annotator_alex.json").read_text(encoding="utf-8")
    )
    record["yoyo_track"] = []
    (out_dir / "records" / f"{record['record_id']}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )

    exit_code = main(
        [
            "merge-cvat",
            str(out_dir),
            record["record_id"],
            str(xml_path),
            "--fps",
            "30",
            "--width",
            "1920",
            "--height",
            "1080",
        ]
    )
    assert exit_code == 0
    updated = json.loads((out_dir / "records" / f"{record['record_id']}.json").read_text())
    assert len(updated["yoyo_track"]) == 1
