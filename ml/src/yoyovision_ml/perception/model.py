"""TinyYoYoNet architecture shared by training and inference."""

from __future__ import annotations

from typing import Any

INPUT_SIZE = 128


def build_tiny_yoyo_net(torch_module: Any) -> Any:
    """Small conv net: 3 conv/pool blocks -> 5-value bbox + confidence head."""
    nn = torch_module.nn

    class TinyYoyoNet(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(64, 5)

        def forward(self, x: Any) -> Any:
            features = self.features(x).flatten(1)
            return self.head(features)

    return TinyYoyoNet()


def import_torch() -> Any:
    from yoyovision_ml.perception.errors import MissingOptionalDependencyError

    try:
        import torch
    except ImportError as exc:
        raise MissingOptionalDependencyError("torch", "torch") from exc
    return torch
