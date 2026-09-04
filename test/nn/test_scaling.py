from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from sdm.nn import GatedLogScale, LogScale, QASSMax
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
@pytest.mark.parametrize(
    "key_len_fn",
    [
        lambda: 4,
        lambda: torch.tensor([[1, 2, 0], [3, 4, 1]]),
    ],
)
def test_log_scale(
    device: torch.device,
    key_len_fn: Callable[[], Tensor | int],
) -> None:
    module = LogScale(num_heads=3, device=device)
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


@withCUDA
@pytest.mark.parametrize(
    "key_len_fn",
    [
        lambda: 7,
        lambda: torch.tensor([[1, 2, 0], [3, 4, 1]]),
    ],
)
def test_gated_log_scale_initialization(
    device: torch.device,
    key_len_fn: Callable[[], Tensor | int],
) -> None:
    module = GatedLogScale(
        channels=2,
        num_heads=3,
        hidden_channels=4,
        device=device,
    )
    query = torch.randn(2, 3, 3, 2, device=device)
    key_len = key_len_fn()
    if isinstance(key_len, Tensor):
        key_len = key_len.to(device)

    out = module(query, key_len=key_len)
    torch.testing.assert_close(
        module.head_scale,
        torch.full((3,), 0.43, device=device),
    )
    expected = LogScale.forward(module, query=query, key_len=key_len)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_gated_log_scale_bounds(device: torch.device) -> None:
    module = GatedLogScale(
        channels=2,
        num_heads=3,
        hidden_channels=4,
        device=device,
    )
    query = torch.ones(2, 3, 3, 2, device=device)
    baseline = LogScale.forward(module, query=query, key_len=7)

    with torch.no_grad():
        gate = module.gate[-1]
        assert isinstance(gate, torch.nn.Linear)
        gate.weight.zero_()
        gate.bias.copy_(torch.tensor([100.0, -100.0], device=device))

    out = module(query, key_len=7)
    factor = out / baseline
    assert (factor >= 0).all()
    assert (factor <= 2).all()
    torch.testing.assert_close(
        factor[..., 0], torch.full_like(factor[..., 0], 2)
    )
    torch.testing.assert_close(
        factor[..., 1], torch.zeros_like(factor[..., 1])
    )
