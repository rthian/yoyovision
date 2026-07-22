from __future__ import annotations

from pathlib import Path

import pytest

from yoyovision_ml.perception.errors import MissingOptionalDependencyError
from yoyovision_ml.perception.overlay import _nearest_ms, render_overlay_video


def test_render_overlay_video_raises_clearly_without_cv2(tmp_path: Path) -> None:
    """`cv2` is not installed in this test environment (confirmed absent),
    so this exercises the required "fail clearly, never silently degrade"
    behaviour rather than needing real video I/O."""
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"not a real video")
    output_path = tmp_path / "overlay.mp4"

    with pytest.raises(MissingOptionalDependencyError) as exc_info:
        render_overlay_video(video_path, output_path, tracks=[])

    assert "cv2" in str(exc_info.value)
    assert "mediapipe" in str(exc_info.value)


def test_nearest_ms_finds_closest_within_tolerance() -> None:
    sorted_ms = [0, 100, 300, 700]
    assert _nearest_ms(sorted_ms, 90) == 100
    assert _nearest_ms(sorted_ms, 110) == 100
    assert _nearest_ms(sorted_ms, 0) == 0


def test_nearest_ms_returns_none_outside_tolerance() -> None:
    sorted_ms = [0, 1000]
    assert _nearest_ms(sorted_ms, 500, tolerance_ms=200) is None


def test_nearest_ms_returns_none_for_empty_list() -> None:
    assert _nearest_ms([], 100) is None


def test_nearest_ms_exact_match() -> None:
    sorted_ms = [0, 50, 100, 150]
    assert _nearest_ms(sorted_ms, 100) == 100
