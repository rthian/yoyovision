from __future__ import annotations

import pytest

from yoyovision_ml.events.config import TrainingConfig
from yoyovision_ml.events.labels import NUM_CLASSES, NUM_OUTCOMES
from yoyovision_ml.events.model import build_model, count_parameters, import_torch

torch = pytest.importorskip("torch")


def _config(**overrides: object) -> TrainingConfig:
    defaults: dict[str, object] = {"hidden_channels": 8, "num_blocks": 2, "kernel_size": 3}
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


def test_import_torch_returns_the_real_torch_module() -> None:
    assert import_torch() is torch


def test_forward_output_has_one_entry_per_head_with_expected_shapes() -> None:
    input_dim = 5
    batch_size, seq_len = 2, 16
    model = build_model(torch, input_dim, _config())
    x = torch.randn(batch_size, seq_len, input_dim)

    output = model(x)

    assert set(output) == {"class_logits", "start_logits", "end_logits", "outcome_logits"}
    assert output["class_logits"].shape == (batch_size, seq_len, NUM_CLASSES)
    assert output["start_logits"].shape == (batch_size, seq_len, NUM_CLASSES)
    assert output["end_logits"].shape == (batch_size, seq_len, NUM_CLASSES)
    assert output["outcome_logits"].shape == (batch_size, seq_len, NUM_OUTCOMES)


def test_forward_preserves_sequence_length_regardless_of_dilation_stack_depth() -> None:
    input_dim = 3
    seq_len = 37  # deliberately not a power of two, to catch off-by-one padding bugs
    model = build_model(torch, input_dim, _config(num_blocks=4))
    x = torch.randn(1, seq_len, input_dim)

    output = model(x)

    for tensor in output.values():
        assert tensor.shape[1] == seq_len


def test_build_model_creates_the_configured_number_of_temporal_blocks() -> None:
    model = build_model(torch, 4, _config(num_blocks=5))
    assert len(model.blocks) == 5


def test_count_parameters_only_counts_trainable_parameters() -> None:
    model = build_model(torch, 4, _config())
    total = count_parameters(model)
    assert total > 0
    assert total == sum(p.numel() for p in model.parameters() if p.requires_grad)

    for param in model.parameters():
        param.requires_grad_(False)
    assert count_parameters(model) == 0


def test_forward_is_deterministic_in_eval_mode_for_the_same_input() -> None:
    model = build_model(torch, 4, _config())
    model.eval()
    x = torch.randn(1, 10, 4)

    with torch.no_grad():
        first = model(x)["class_logits"]
        second = model(x)["class_logits"]

    assert torch.equal(first, second)


def test_different_seeds_produce_different_initial_weights() -> None:
    # `input_norm`'s GroupNorm weight/bias are constant-initialized (1s/0s),
    # so compare a conv layer's weights instead, which torch initializes
    # from a seed-dependent random distribution.
    torch.manual_seed(0)
    model_a = build_model(torch, 4, _config())
    torch.manual_seed(1)
    model_b = build_model(torch, 4, _config())

    assert not torch.equal(model_a.blocks[0].conv1.weight, model_b.blocks[0].conv1.weight)
