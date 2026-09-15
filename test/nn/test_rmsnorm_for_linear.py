import copy
from typing import cast

import pytest
import torch
from torch.nn import Linear, RMSNorm, Sequential

from sdm.nn._rmsnorm_for_linear import _RMSNormForLinear
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("rows", "channels"), [(1, 128), (513, 256), (1024, 512), (4096, 128)]
)
def test_rmsnorm_linear(
    device: torch.device,
    dtype: torch.dtype,
    autocast_dtype: torch.dtype,
    rows: int,
    channels: int,
) -> None:
    native = Sequential(
        RMSNorm(channels, device=device),
        Linear(channels, channels, device=device),
    ).eval()
    with torch.no_grad():
        weight = cast(RMSNorm, native[0]).weight
        assert weight is not None
        weight.uniform_(-2, 2)
    fused = copy.deepcopy(native)
    fused[0] = _RMSNormForLinear(channels, device=device)
    fused.load_state_dict(native.state_dict())
    fused.eval()
    x = torch.randn(2, rows, 2 * channels, dtype=dtype, device=device)[
        ..., ::2
    ]
    x[0, 0] = 0
    if rows > 1:
        x[0, 1, 0] = float("inf")
    with (
        torch.inference_mode(),
        torch.autocast(
            device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ),
    ):
        if device.type == "cpu":
            x = x.float()
        torch.testing.assert_close(
            fused(x), native(x), rtol=0, atol=0, equal_nan=True
        )


@withCUDA
def test_rmsnorm_linear_gradients(device: torch.device) -> None:
    native = RMSNorm(128, device=device).eval()
    fused = _RMSNormForLinear(128, device=device).eval()
    fused.load_state_dict(native.state_dict())
    x = torch.randn(2, 17, 128, device=device, requires_grad=True)
    expected = native(x)
    actual = fused(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.sum().backward()
    assert x.grad is not None
    grad = x.grad.clone()
    x.grad = None
    actual.sum().backward()
    torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)
    torch.testing.assert_close(
        fused.weight.grad, native.weight.grad, rtol=0, atol=0
    )
