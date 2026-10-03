# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.models.kumo.timeseries.anomaly.block import ResidualBlock
from sdm.models.kumo.timeseries.anomaly.denoiser import DiffusionDenoiser
from sdm.testing import withCUDA


def reference_attention(
    layer: torch.nn.TransformerEncoderLayer, x: Tensor
) -> Tensor:
    # Explicit post-norm reference, without the encoder's fused fast path.
    attn, _ = layer.self_attn(x, x, x, need_weights=False)
    x = layer.norm1(x + attn)
    return layer.norm2(x + layer.linear2(F.gelu(layer.linear1(x))))


def reference_block(
    block: ResidualBlock,
    x: Tensor,
    side_info: Tensor,
    diffusion_embedding: Tensor,
    strategy_embedding: Tensor,
) -> tuple[Tensor, Tensor]:
    # Keep the released channels-first/flattened layout as an independent
    # reference for the port's channels-last temporal and feature operations.
    batch, channels, features, length = x.shape
    y = x.reshape(batch, channels, features * length)
    y = y + block.diffusion_projection(diffusion_embedding).unsqueeze(-1)
    y = y + block.strategy_projection(strategy_embedding).unsqueeze(-1)
    if length != 1:
        y = y.reshape(batch, channels, features, length)
        y = y.permute(0, 2, 1, 3).reshape(batch * features, channels, length)
        y = reference_attention(block.time_layer, y.transpose(1, 2))
        y = block.norm_after_time(y).transpose(1, 2)
        y = y.reshape(batch, features, channels, length).permute(0, 2, 1, 3)
        y = y.reshape(batch, channels, features * length)
    if features != 1:
        y = y.reshape(batch, channels, features, length)
        y = y.permute(0, 3, 1, 2).reshape(batch * length, channels, features)
        y = reference_attention(block.feature_layer, y.transpose(1, 2))
        y = block.norm_after_feature(y).transpose(1, 2)
        y = y.reshape(batch, length, channels, features).permute(0, 2, 3, 1)
        y = y.reshape(batch, channels, features * length)
    y = block.mid_projection(y) + block.cond_projection(side_info.flatten(2))
    gate, value = y.chunk(2, dim=1)
    y = gate.sigmoid() * value.tanh()
    y = block.norm_after_gate(y.transpose(1, 2)).transpose(1, 2)
    residual, skip = block.output_projection(y).chunk(2, dim=1)
    return (x + residual.reshape_as(x)) / math.sqrt(2), skip.reshape_as(x)


@withCUDA
@pytest.mark.parametrize(
    ("features", "length"), [(3, 5), (1, 5), (3, 1), (1, 1)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_residual_block(
    device: torch.device,
    features: int,
    length: int,
    dtype: torch.dtype,
) -> None:
    block = ResidualBlock(
        channels=8,
        side_channels=6,
        embedding_channels=8,
        num_heads=2,
        device=device,
        dtype=dtype,
    ).eval()
    x = torch.randn(2, 8, features, length, device=device, dtype=dtype)
    side = torch.randn(2, 6, features, length, device=device, dtype=dtype)
    diffusion = torch.randn(1, 8, device=device, dtype=dtype)
    strategy = torch.randn(2, 8, device=device, dtype=dtype)
    with torch.inference_mode():
        actual = block(x, side, diffusion, strategy)
        expected = reference_block(block, x, side, diffusion, strategy)
    for result, reference in zip(actual, expected, strict=True):
        assert result.shape == x.shape
        assert result.dtype == dtype
        torch.testing.assert_close(result, reference)


@withCUDA
@pytest.mark.parametrize("input_channels", [1, 2])
@pytest.mark.parametrize(
    ("features", "length"), [(3, 5), (1, 5), (3, 1), (1, 1)]
)
@pytest.mark.parametrize("shared_conditioning", [False, True])
def test_denoiser(
    device: torch.device,
    input_channels: int,
    features: int,
    length: int,
    shared_conditioning: bool,
) -> None:
    model = DiffusionDenoiser(
        channels=8,
        side_channels=6,
        num_steps=10,
        num_layers=2,
        num_heads=2,
        embedding_channels=8,
        input_channels=input_channels,
        device=device,
    ).eval()
    x = torch.randn(2, input_channels, features, length, device=device)
    side = torch.randn(2, 6, features, length, device=device)
    step = torch.tensor([3] if shared_conditioning else [3, 7], device=device)
    strategy = torch.tensor(
        [0] if shared_conditioning else [0, 1], device=device
    )
    with torch.inference_mode():
        actual = model(x, side, step, strategy)
        y = model.input_projection(x.flatten(2)).relu()
        y = y.reshape(2, 8, features, length)
        diffusion_embedding = model.diffusion_embedding(step)
        strategy_embedding = model.strategy_embedding(strategy)
        skips = []
        for block in model.residual_layers:
            y, skip = reference_block(
                block=cast(ResidualBlock, block),
                x=y,
                side_info=side,
                diffusion_embedding=diffusion_embedding,
                strategy_embedding=strategy_embedding,
            )
            skips.append(skip)
        expected = torch.stack(skips).sum(dim=0) / math.sqrt(len(skips))
        expected = model.norm_after_skip(expected.flatten(2).transpose(1, 2))
        expected = model.output_projection1(expected.transpose(1, 2))
        expected = model.norm_after_proj1(expected.transpose(1, 2))
        expected = model.output_projection2(expected.transpose(1, 2).relu())
        expected = expected.reshape(2, features, length)
        assert actual.shape == (2, features, length)
        assert actual.isfinite().all()
        torch.testing.assert_close(actual, expected)
        # Independent windows must not attend across batch elements.
        individual = torch.cat(
            [
                model(
                    x=x[i : i + 1],
                    side_info=side[i : i + 1],
                    step=step if shared_conditioning else step[i : i + 1],
                    strategy=strategy
                    if shared_conditioning
                    else strategy[i : i + 1],
                )
                for i in range(2)
            ]
        )
        torch.testing.assert_close(actual, individual)


@withCUDA
def test_denoiser_backward_and_state_dict(device: torch.device) -> None:
    kwargs: dict[str, Any] = {
        "channels": 8,
        "side_channels": 6,
        "num_steps": 10,
        "num_layers": 2,
        "num_heads": 2,
        "embedding_channels": 8,
        "device": device,
        "dtype": torch.float64,
    }
    model = DiffusionDenoiser(**kwargs).eval()
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
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    assert side.grad is not None
    assert side.grad.abs().sum() > 0
    restored = DiffusionDenoiser(**kwargs).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x, side, step, strategy), out)


@withCUDA
def test_denoiser_dropout(device: torch.device) -> None:
    model = DiffusionDenoiser(
        channels=8,
        side_channels=6,
        num_steps=10,
        num_layers=2,
        num_heads=2,
        embedding_channels=8,
        dropout=1.0,
        device=device,
    )
    x = torch.randn(2, 2, 3, 5, device=device)
    side = torch.randn(2, 6, 3, 5, device=device)
    step = torch.tensor([2, 7], device=device)
    strategy = torch.tensor([0, 1], device=device)
    training = model(x, side, step, strategy)
    assert training.isfinite().all()
    model.eval()
    evaluation = model(x, side, step, strategy)
    assert not torch.allclose(training, evaluation)
    torch.testing.assert_close(model(x, side, step, strategy), evaluation)
