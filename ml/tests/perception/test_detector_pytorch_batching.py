"""Tests for `PyTorchYoyoDetector`'s bounded batching (Prompt F: "Add
bounded batching"). `torch` is installed in this dev environment (unlike
`test_detectors_optional_deps.py`'s assumptions, written when it was not),
so these exercise a genuine forward pass against a real (tiny, untrained)
checkpoint rather than only the missing-dependency error paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from yoyovision_ml.interfaces import FrameRef

torch = pytest.importorskip("torch")


def _write_checkpoint(path: Path) -> None:
    from yoyovision_ml.perception.model import build_tiny_yoyo_net

    net = build_tiny_yoyo_net(torch)
    torch.save({"state_dict": net.state_dict(), "model_version": "test-checkpoint-v1"}, path)


def _frames(count: int) -> list[FrameRef]:
    return [
        FrameRef(frame_ms=i * 33, array=np.random.default_rng(i).integers(0, 255, (64, 64, 3), dtype=np.uint8))
        for i in range(count)
    ]


def test_predict_batches_frames_into_chunks_of_batch_size(tmp_path: Path) -> None:
    from yoyovision_ml.perception.detector_pytorch import PyTorchYoyoDetector

    checkpoint = tmp_path / "weights.pt"
    _write_checkpoint(checkpoint)
    detector = PyTorchYoyoDetector(weights_path=checkpoint, batch_size=2)

    forward_call_sizes: list[int] = []
    original_model = detector._model  # type: ignore[attr-defined]
    original_forward = original_model.forward

    def counting_forward(x: object) -> object:
        forward_call_sizes.append(x.shape[0])  # type: ignore[union-attr]
        return original_forward(x)

    original_model.forward = counting_forward  # type: ignore[method-assign]

    detections = detector.predict(_frames(5))

    assert len(detections) == 5
    assert forward_call_sizes == [2, 2, 1], "5 frames at batch_size=2 must run in 3 bounded chunks"
    assert [d.frame_ms for d in detections] == [0, 33, 66, 99, 132]


def test_predict_preserves_order_with_a_single_batch(tmp_path: Path) -> None:
    from yoyovision_ml.perception.detector_pytorch import PyTorchYoyoDetector

    checkpoint = tmp_path / "weights.pt"
    _write_checkpoint(checkpoint)
    detector = PyTorchYoyoDetector(weights_path=checkpoint, batch_size=8)

    detections = detector.predict(_frames(3))

    assert [d.frame_ms for d in detections] == [0, 33, 66]
    for detection in detections:
        assert detection.model_version == "test-checkpoint-v1+torch" + torch.__version__


def test_predict_skips_frames_with_no_array(tmp_path: Path) -> None:
    from yoyovision_ml.perception.detector_pytorch import PyTorchYoyoDetector

    checkpoint = tmp_path / "weights.pt"
    _write_checkpoint(checkpoint)
    detector = PyTorchYoyoDetector(weights_path=checkpoint, batch_size=4)

    frames = _frames(2) + [FrameRef(frame_ms=999, array=None)]
    detections = detector.predict(frames)

    assert [d.frame_ms for d in detections] == [0, 33]


def test_batch_size_must_be_positive(tmp_path: Path) -> None:
    from yoyovision_ml.perception.detector_pytorch import PyTorchYoyoDetector

    checkpoint = tmp_path / "weights.pt"
    _write_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="batch_size"):
        PyTorchYoyoDetector(weights_path=checkpoint, batch_size=0)
