from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from sdm.nn import PerHeadLogNScale, QASSMax
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


@withCUDA
def test_per_head_log_n_scale_initialization(device: torch.device) -> None:
    module = PerHeadLogNScale(
        num_heads=3,
        device=device,
        dtype=torch.float64,
    )

    assert module.head_scale.shape == (3,)
    assert isinstance(module.head_scale, torch.nn.Parameter)
    assert module.head_scale.requires_grad
    assert module.head_scale.dtype == torch.float64
    assert module.head_scale.device == device
    torch.testing.assert_close(
        module.head_scale,
        torch.full((3,), 0.43, dtype=torch.float64, device=device),
    )


@withCUDA
@pytest.mark.parametrize(
    "key_len_fn",
    [
        lambda: 4,
        lambda: 0,
        lambda: torch.tensor([[1, 2, 0], [-3, 4, 1]]),
        lambda: torch.tensor([[4], [1]], dtype=torch.int32),
    ],
)
def test_per_head_log_n_scale(
    device: torch.device,
    key_len_fn: Callable[[], Tensor | int],
) -> None:
    module = PerHeadLogNScale(num_heads=3, device=device)
    with torch.no_grad():
        module.head_scale.copy_(torch.tensor([0.5, 1.0, 1.5], device=device))

    query = torch.arange(1, 37, dtype=torch.float32, device=device).reshape(
        2, 3, 3, 2
    )
    key_len = key_len_fn()
    if isinstance(key_len, Tensor):
        key_len = key_len.to(device)
        expected_key_len = key_len.float().broadcast_to((2, 3))
    else:
        expected_key_len = torch.full((2, 3), key_len, device=device)

    out = module(query, key_len=key_len)

    expected_scale = expected_key_len.clamp(min=1.0).log()[..., None, None]
    expected_head_scale = module.head_scale[None, None, :, None]
    torch.testing.assert_close(
        out,
        query * expected_scale * expected_head_scale,
    )
    assert out.dtype == query.dtype
    assert out.device == query.device


@withCUDA
@pytest.mark.parametrize(
    "key_len_fn",
    [
        lambda: 257,
        lambda: torch.tensor([[257]], dtype=torch.int32),
    ],
)
def test_per_head_log_n_scale_uses_fp32_log(
    device: torch.device,
    key_len_fn: Callable[[], Tensor | int],
) -> None:
    module = PerHeadLogNScale(num_heads=1, device=device)
    with torch.no_grad():
        module.head_scale.fill_(1.0)
    query = torch.ones((1, 1, 1, 1), dtype=torch.bfloat16, device=device)
    key_len = key_len_fn()
    if isinstance(key_len, Tensor):
        key_len = key_len.to(device)

    out = module(query, key_len=key_len)

    expected = (
        torch.tensor(257, dtype=torch.float32, device=device)
        .log()
        .to(query.dtype)
    )
    torch.testing.assert_close(out, expected.expand_as(query), rtol=0, atol=0)
    assert out.dtype == query.dtype
    assert out.device == query.device
