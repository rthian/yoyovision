"""The temporal trick-event model architecture: a modest TCN encoder plus
three per-frame prediction heads.

Per Prompt C's "Implement a modest and reproducible baseline before using a
large video transformer" -- this is a small dilated-convolution stack, not a
transformer. `torch`/`torch.nn` are imported lazily (only inside
`import_torch`/`build_model`), mirroring `perception/detector_pytorch.py`'s
pattern, so importing this module never requires the optional `torch` extra;
only actually building/training/running the model does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yoyovision_ml.events.config import TrainingConfig
from yoyovision_ml.events.labels import NUM_CLASSES, NUM_OUTCOMES
from yoyovision_ml.perception.errors import MissingOptionalDependencyError

if TYPE_CHECKING:
    import torch.nn as nn


def import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise MissingOptionalDependencyError("torch", "torch") from exc
    return torch


def _build_temporal_conv_block(
    torch_module: Any,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dilation: int,
    dropout: float,
) -> Any:
    nn = torch_module.nn

    class TemporalConvBlock(nn.Module):  # type: ignore[misc, name-defined]
        """One dilated causal-ish conv1d residual block: `Conv1d -> GroupNorm
        -> ReLU -> Dropout`, twice, plus a residual connection (with a 1x1
        projection when channel counts differ). `padding` is chosen so the
        block never changes the sequence length -- every block in the stack
        stays aligned to the same `frame_ms` grid the input arrived on."""

        def __init__(self) -> None:
            super().__init__()
            padding = dilation * (kernel_size - 1) // 2
            self.conv1 = nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
            )
            self.norm1 = nn.GroupNorm(1, out_channels)
            self.conv2 = nn.Conv1d(
                out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
            )
            self.norm2 = nn.GroupNorm(1, out_channels)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.ReLU()
            self.residual_proj = (
                nn.Conv1d(in_channels, out_channels, 1)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x: Any) -> Any:
            residual = self.residual_proj(x)
            out = self.activation(self.norm1(self.conv1(x)))
            out = self.dropout(out)
            out = self.activation(self.norm2(self.conv2(out)))
            out = self.dropout(out)
            return self.activation(out + residual)

    return TemporalConvBlock()


def build_model(torch_module: Any, input_dim: int, config: TrainingConfig) -> Any:
    """Builds a fresh, untrained `TrickEventTCN` for `input_dim` input
    feature columns (varies with `config.feature_subset` -- see
    `labels.FEATURE_SUBSETS`)."""
    nn = torch_module.nn

    class TrickEventTCN(nn.Module):  # type: ignore[misc, name-defined]
        """Dilated-conv1d encoder (`config.num_blocks` blocks, exponentially
        increasing dilation) shared by three per-frame heads:

        - `class_logits`  `(B, T, NUM_CLASSES)`  -- multi-label classification
        - `start_logits`, `end_logits`  `(B, T, NUM_CLASSES)`  -- boundary head
        - `outcome_logits`  `(B, T, NUM_OUTCOMES)`  -- outcome head

        All heads read the same shared encoder output; nothing here is a
        validated architecture -- it is intentionally small and simple so a
        training run is fast, reproducible, and easy to reason about, per
        Prompt C's "modest and reproducible baseline" requirement.
        """

        def __init__(self) -> None:
            super().__init__()
            self.input_norm = nn.GroupNorm(1, input_dim)
            blocks = []
            in_channels = input_dim
            for block_idx in range(config.num_blocks):
                dilation = 2**block_idx
                blocks.append(
                    _build_temporal_conv_block(
                        torch_module,
                        in_channels,
                        config.hidden_channels,
                        config.kernel_size,
                        dilation,
                        config.dropout,
                    )
                )
                in_channels = config.hidden_channels
            self.blocks = nn.ModuleList(blocks)

            self.class_head = nn.Linear(config.hidden_channels, NUM_CLASSES)
            self.start_head = nn.Linear(config.hidden_channels, NUM_CLASSES)
            self.end_head = nn.Linear(config.hidden_channels, NUM_CLASSES)
            self.outcome_head = nn.Linear(config.hidden_channels, NUM_OUTCOMES)

        def forward(self, x: Any) -> dict[str, Any]:
            """`x`: `(B, T, input_dim)` -> dict of `(B, T, ...)` logit tensors."""
            hidden = x.transpose(1, 2)  # (B, input_dim, T)
            hidden = self.input_norm(hidden)
            for block in self.blocks:
                hidden = block(hidden)
            hidden = hidden.transpose(1, 2)  # (B, T, hidden_channels)
            return {
                "class_logits": self.class_head(hidden),
                "start_logits": self.start_head(hidden),
                "end_logits": self.end_head(hidden),
                "outcome_logits": self.outcome_head(hidden),
            }

    return TrickEventTCN()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
