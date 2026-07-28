import pytest
import torch

from sdm.nn.resolver import normalization_resolver


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

    with pytest.raises(ValueError, match="resolve normalization 'linear'"):
        normalization_resolver("linear")


def test_normalization_resolver_callable() -> None:
    # A callable is called with the given arguments on every resolution.
    module = normalization_resolver(
        torch.nn.LayerNorm,
        8,
        eps=1e-6,
        dtype=torch.float64,
    )
    assert isinstance(module, torch.nn.LayerNorm)
    assert module.normalized_shape == (8,)
    assert module.eps == 1e-6
    assert next(module.parameters()).dtype == torch.float64

    other = normalization_resolver(torch.nn.LayerNorm, 8)
    assert other is not module


def test_normalization_resolver_module() -> None:
    # A module instance is returned unchanged.
    module = torch.nn.LayerNorm(8)
    assert normalization_resolver(module) is module
