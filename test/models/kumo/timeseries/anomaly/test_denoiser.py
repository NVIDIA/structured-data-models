# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.models.kumo.timeseries.anomaly.denoiser import DiffusionDenoiser
from sdm.testing import withCUDA


def reference_denoiser(
    model: DiffusionDenoiser,
    x: Tensor,
    side: Tensor,
    step: Tensor,
    strategy: Tensor,
) -> Tensor:
    """Compose the tested blocks with pointwise linear projections."""
    x = x.permute(0, 2, 3, 1)
    projection = model.input_projection
    x = F.linear(x, projection.weight.squeeze(-1), projection.bias).relu()
    x = x.permute(0, 3, 1, 2)
    diffusion = model.diffusion_embedding(step)
    strategy_embedding = model.strategy_embedding(strategy)
    skips = []
    for block in model.residual_layers:
        x, skip = block(x, side, diffusion, strategy_embedding)
        skips.append(skip.permute(0, 2, 3, 1))
    x = sum(skips) / math.sqrt(len(skips))
    x = model.norm_after_skip(x)
    projection = model.output_projection1
    x = F.linear(x, projection.weight.squeeze(-1), projection.bias)
    x = model.norm_after_proj1(x).relu()
    projection = model.output_projection2
    x = F.linear(x, projection.weight.squeeze(-1), projection.bias)
    return x.squeeze(-1)


@withCUDA
@pytest.mark.parametrize(
    (
        "input_channels",
        "num_layers",
        "features",
        "length",
        "step_batch",
        "strategy_batch",
    ),
    [
        (2, 1, 3, 5, 1, 2),
        (1, 3, 3, 5, 2, 1),
        (2, 2, 1, 1, 2, 2),
    ],
)
def test_denoiser(
    device: torch.device,
    input_channels: int,
    features: int,
    length: int,
    num_layers: int,
    step_batch: int,
    strategy_batch: int,
) -> None:
    model = DiffusionDenoiser(
        channels=8,
        side_channels=6,
        num_steps=10,
        num_layers=num_layers,
        num_heads=2,
        embedding_channels=12,
        input_channels=input_channels,
        device=device,
    ).eval()
    x = torch.randn(2, input_channels, features, length, device=device)
    side = torch.randn(2, 6, features, length, device=device)
    step = torch.tensor([0, 9], device=device)[:step_batch]
    strategy = torch.tensor([0, 1], device=device)[:strategy_batch]
    with torch.inference_mode():
        actual = model(x, side, step, strategy)
        expected = reference_denoiser(model, x, side, step, strategy)
        assert actual.shape == (2, features, length)
        assert actual.dtype == x.dtype
        assert actual.device == device
        assert actual.isfinite().all()
        torch.testing.assert_close(actual, expected)


@withCUDA
def test_denoiser_grad(device: torch.device) -> None:
    model = DiffusionDenoiser(
        channels=8,
        side_channels=6,
        num_steps=10,
        num_layers=2,
        num_heads=2,
        embedding_channels=12,
        device=device,
        dtype=torch.float64,
    ).eval()
    x = torch.randn(
        2, 2, 3, 5, device=device, dtype=torch.float64, requires_grad=True
    )
    side = torch.randn(
        2, 6, 3, 5, device=device, dtype=torch.float64, requires_grad=True
    )
    step = torch.tensor([0, 9], device=device)
    strategy = torch.tensor([0, 1], device=device)
    out = model(x, side, step, strategy)
    assert out.dtype == torch.float64
    out.square().mean().backward()
    for value in (x, side, *model.parameters()):
        assert value.grad is not None
        assert value.grad.isfinite().all()
    for value in (x, side):
        assert value.grad is not None
        assert value.grad.abs().sum() > 0
