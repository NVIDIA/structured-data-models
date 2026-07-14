import pytest
import torch
from sdm.nn import normalization_resolver


def test_normalization_resolver() -> None:
    module = normalization_resolver(
        "layer_norm",
        8,
        eps=1e-6,
        dtype=torch.float64,
    )
    assert isinstance(module, torch.nn.LayerNorm)
    assert next(module.parameters()).dtype == torch.float64
    assert module.eps == 1e-6

    module = normalization_resolver(
        "batch_1d",
        8,
        eps=1e-6,
        dtype=torch.float64,
    )
    assert isinstance(module, torch.nn.BatchNorm1d)
    assert next(module.parameters()).dtype == torch.float64
    assert module.eps == 1e-6

    # No-op for a Module instance.
    module = torch.nn.LayerNorm(8)
    assert normalization_resolver(module) is module

    with pytest.raises(
        ValueError,
        match="Could not resolve normalization 'linear'",
    ):
        normalization_resolver("linear", 8, 8)
