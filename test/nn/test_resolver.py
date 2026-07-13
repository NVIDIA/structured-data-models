from typing import Any, cast

import pytest
import torch
from sdm.nn import normalization_resolver


@pytest.mark.parametrize(
    ("query", "normalization_cls"),
    [
        ("layer_norm", torch.nn.LayerNorm),
        ("batch_1d", torch.nn.BatchNorm1d),
    ],
)
def test_normalization_resolver(
    query: str,
    normalization_cls: type[torch.nn.Module],
) -> None:
    module = normalization_resolver(
        query,
        8,
        eps=1e-6,
        dtype=torch.float64,
    )

    assert isinstance(module, normalization_cls)
    assert next(module.parameters()).dtype == torch.float64
    normalization = cast(torch.nn.LayerNorm | torch.nn.BatchNorm1d, module)
    assert normalization.eps == 1e-6


def test_normalization_resolver_module_passthrough() -> None:
    module = torch.nn.LayerNorm(8)

    assert normalization_resolver(module) is module


def test_normalization_resolver_errors() -> None:
    with pytest.raises(
        TypeError,
        match=r"`query` must be a string or torch\.nn\.Module",
    ):
        normalization_resolver(cast(Any, 1))

    with pytest.raises(
        ValueError,
        match="Could not resolve normalization 'linear'",
    ):
        normalization_resolver("linear", 8, 8)
