"""Tiny numpy-only image resize/format helpers shared by the real detector
adapters (`detector_pytorch.py`, `detector_onnx.py`) and `overlay.py`.

Deliberately does not depend on OpenCV/PIL: `preprocessing.extract_frames`
already only produces `FrameRef.array` when OpenCV happens to be installed,
but the *detector* adapters here should not additionally require it just to
resize a frame for their own fixed input size. Nearest-neighbor is
sufficient for a placeholder architecture; a real trained model's own
preprocessing pipeline would specify whatever resampling it was trained with.
"""

from __future__ import annotations

import numpy as np


def resize_nearest(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize of an `(H, W, C)` array to `(size[1], size[0], C)`."""
    target_w, target_h = size
    src_h, src_w = array.shape[0], array.shape[1]
    row_idx = (np.arange(target_h) * src_h / target_h).astype(np.intp)
    col_idx = (np.arange(target_w) * src_w / target_w).astype(np.intp)
    resized: np.ndarray = array[row_idx][:, col_idx]
    return resized


def hwc_uint8_to_chw_float01(array: np.ndarray) -> np.ndarray:
    """`(H, W, 3)` uint8 -> `(3, H, W)` float32 in `[0, 1]`."""
    normalized = array.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))
