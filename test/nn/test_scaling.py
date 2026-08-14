from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from sdm.nn import QASSMax
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    "key_len_fn",
    [
        lambda: 4,
        lambda: torch.tensor([[1, 2, 0], [3, 4, 1]]),
        lambda: torch.tensor([[4], [1]], dtype=torch.int32),
    ],
)
def test_qassmax(
    device: torch.device,
    key_len_fn: Callable[[], Tensor | int],
) -> None:
    channels = 2
    num_heads = 3
    module = QASSMax(
        channels=channels,
        num_heads=num_heads,
        hidden_channels=4,
        device=device,
    )

    query = torch.randn(2, 3, num_heads, channels, device=device)

    key_len = key_len_fn()
    if isinstance(key_len, Tensor):
        key_len = key_len.to(device)

    out = module(query, key_len=key_len)
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device
