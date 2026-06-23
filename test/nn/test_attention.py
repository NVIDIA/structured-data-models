from collections.abc import Callable

import pytest
import torch
from schemafm.nn.attention import QASSMax
from torch import Tensor


@pytest.mark.parametrize(
    "key_len_fn",
    [
        lambda: 4,
        lambda: torch.tensor([[1, 2, 0], [3, 4, 1]]),
        lambda: torch.tensor([[4], [1]], dtype=torch.int32),
    ],
)
def test_qassmax(key_len_fn: Callable[[], Tensor | int]) -> None:
    channels = 2
    num_heads = 3
    module = QASSMax(channels=channels, num_heads=num_heads, hidden_channels=4)

    query = torch.randn(2, 3, num_heads, channels)

    out = module(query, key_len_fn())
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device
