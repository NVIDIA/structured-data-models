# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.models.kumo.timeseries.forecast.normalization import RevIN
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    ("dtype", "affine", "num_features", "masked"),
    [
        (torch.float32, False, 1, False),
        (torch.float64, True, 2, True),
        (torch.float16, False, 1, True),
        (torch.bfloat16, True, 1, True),
    ],
)
def test_revin(
    device: torch.device,
    dtype: torch.dtype,
    affine: bool,
    num_features: int,
    masked: bool,
) -> None:
    """Match observed population statistics and invert explicit state."""
    module = RevIN(
        num_features=num_features, affine=affine, device=device, dtype=dtype
    )
    if affine:
        with torch.no_grad():
            module.affine_weight.fill_(1.5)
            module.affine_bias.fill_(0.25)
    x = torch.tensor(
        data=[[[1.0, 3.0, 5.0, 97.0], [-5.0, 0.0, 5.0, -53.0]]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, True, False]], device=device)
    normalized, state = module(x, mask if masked else None)
    observed = x[..., :3] if masked else x
    mean = observed.mean(dim=-1, keepdim=True)
    stdev = (
        observed.var(dim=-1, correction=0, keepdim=True).sqrt() + module.eps
    )
    expected = (x - mean) / stdev
    if affine:
        expected = expected * module.affine_weight + module.affine_bias
    torch.testing.assert_close(state.mean, mean)
    torch.testing.assert_close(state.stdev, stdev)
    torch.testing.assert_close(normalized, expected)
    assert not state.mean.requires_grad
    assert not state.stdev.requires_grad
    assert normalized.dtype == dtype
    assert normalized.device == device

    # A later call must not overwrite the statistics used by this inverse.
    module(x + 10)
    torch.testing.assert_close(module.inverse(normalized, state), x)
    forecast = normalized[..., :2]
    torch.testing.assert_close(module.inverse(forecast, state), x[..., :2])
    if dtype == torch.float64:
        normalized.sum().backward()
        weight = module.affine_weight if affine else 1.0
        torch.testing.assert_close(x.grad, (weight / stdev).expand_as(x))


@withCUDA
@pytest.mark.parametrize("affine", [False, True])
def test_revin_missing_and_constant(
    device: torch.device, affine: bool
) -> None:
    """Keep missing values and use neutral statistics for unobserved series."""
    module = RevIN(num_features=1, affine=affine, device=device)
    x = torch.tensor(
        data=[
            [[float("nan")] * 4, [1.0, float("nan"), 3.0, 5.0]],
            [[2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0]],
            [[3.0] * 4, [-2.0] * 4],
        ],
        device=device,
    )
    mask = torch.tensor(
        data=[[True] * 4, [False] * 4, [True] * 4], device=device
    )
    normalized, state = module(x, mask)
    mean = x.new_tensor([[[0.0], [3.0]], [[0.0], [0.0]], [[3.0], [-2.0]]])
    stdev = torch.ones_like(mean)
    stdev[0, 1] = (8.0 / 3.0) ** 0.5 + module.eps
    stdev[2] = module.eps
    torch.testing.assert_close(state.mean, mean)
    torch.testing.assert_close(state.stdev, stdev)
    torch.testing.assert_close(normalized[1], x[1])
    torch.testing.assert_close(normalized[2], torch.zeros_like(x[2]))
    assert torch.equal(normalized.isnan(), x.isnan())
    torch.testing.assert_close(
        actual=module.inverse(normalized, state), expected=x, equal_nan=True
    )


def test_revin_meta_device() -> None:
    """Construct affine normalization without materializing model weights."""
    with torch.device("meta"):
        module = RevIN(num_features=2, affine=True)
        x = torch.empty(3, 2, 8)
        normalized, state = module(x)
        restored = module.inverse(normalized, state)
    assert restored.shape == x.shape
    assert restored.device.type == "meta"
    assert all(
        parameter.device.type == "meta" for parameter in module.parameters()
    )
