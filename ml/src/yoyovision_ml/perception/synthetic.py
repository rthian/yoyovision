"""Synthetic frame samples for yo-yo detector training smoke tests."""

from __future__ import annotations

import hashlib
import struct

import numpy as np

from yoyovision_ml.perception.types import DetectorTrainingSample

_SYNTHETIC_TOOL = "yoyovision_ml.perception.synthetic"


def _rng(seed: int, *parts: object) -> np.random.Generator:
    digest = hashlib.sha256(struct.pack(">q", seed) + repr(parts).encode()).digest()
    subseed = int.from_bytes(digest[:8], "big") % (2**32 - 1)
    return np.random.default_rng(subseed)


def generate_synthetic_detector_samples(
    *,
    seed: int = 42,
    num_players: int = 4,
    frames_per_player: int = 8,
    image_size: int = 64,
) -> list[DetectorTrainingSample]:
    samples: list[DetectorTrainingSample] = []
    for player_idx in range(num_players):
        player_id = f"synthetic-player-{player_idx}"
        video_id = f"synthetic-video-{player_idx}"
        rng = _rng(seed, player_id, video_id)
        for frame_idx in range(frames_per_player):
            frame_ms = frame_idx * 200
            cx = float(0.2 + 0.6 * rng.random())
            cy = float(0.2 + 0.6 * rng.random())
            size = 0.08
            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            px = int(cx * image_size)
            py = int(cy * image_size)
            radius = max(2, int(size * image_size / 2))
            y0, y1 = max(0, py - radius), min(image_size, py + radius)
            x0, x1 = max(0, px - radius), min(image_size, px + radius)
            image[y0:y1, x0:x1] = (40, 200, 40)
            samples.append(
                DetectorTrainingSample(
                    video_id=video_id,
                    player_id=player_id,
                    frame_ms=frame_ms,
                    image=image,
                    target_bbox=(cx - size / 2, cy - size / 2, size, size),
                    visible=True,
                )
            )
    return samples
