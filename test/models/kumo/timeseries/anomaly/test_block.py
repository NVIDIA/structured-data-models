# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.models.kumo.timeseries.anomaly.block import ResidualBlock
from sdm.testing import withCUDA


def reference_attention(
    layer: torch.nn.TransformerEncoderLayer, x: Tensor
) -> Tensor:
    """Evaluate post-norm attention without the encoder's fused fast path."""
    attn, _ = layer.self_attn(x, x, x, need_weights=False)
    x = layer.norm1(x + layer.dropout1(attn))
    hidden = layer.linear2(layer.dropout(F.gelu(layer.linear1(x))))
    return layer.norm2(x + layer.dropout2(hidden))


def reference_block(
    block: ResidualBlock,
    x: Tensor,
    side_info: Tensor,
    diffusion_embedding: Tensor,
    strategy_embedding: Tensor,
) -> tuple[Tensor, Tensor]:
    """Use the upstream channels-first layout as a numerical reference."""
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
@pytest.mark.parametrize("embedding_channels", [8, 12])
@pytest.mark.parametrize(
    ("diffusion_batch", "strategy_batch"), [(1, 2), (2, 1)]
)
@pytest.mark.parametrize("training", [False, True])
def test_residual_block(
    device: torch.device,
    features: int,
    length: int,
    dtype: torch.dtype,
    embedding_channels: int,
    diffusion_batch: int,
    strategy_batch: int,
    training: bool,
) -> None:
    block = ResidualBlock(
        channels=8,
        side_channels=6,
        embedding_channels=embedding_channels,
        num_heads=2,
        dropout=1.0,
        device=device,
        dtype=dtype,
    ).train(training)
    x = torch.randn(2, 8, features, length, device=device, dtype=dtype)
    side = torch.randn(2, 6, features, length, device=device, dtype=dtype)
    diffusion = torch.randn(
        diffusion_batch, embedding_channels, device=device, dtype=dtype
    )
    strategy = torch.randn(
        strategy_batch, embedding_channels, device=device, dtype=dtype
    )
    with torch.inference_mode():
        actual = block(x, side, diffusion, strategy)
        expected = reference_block(block, x, side, diffusion, strategy)
    for result, reference in zip(actual, expected, strict=True):
        assert result.shape == x.shape
        assert result.dtype == dtype
        assert result.device == device
        torch.testing.assert_close(result, reference)


@withCUDA
def test_residual_block_grad_and_state_dict(device: torch.device) -> None:
    block = ResidualBlock(
        channels=8,
        side_channels=6,
        embedding_channels=12,
        num_heads=2,
        device=device,
        dtype=torch.float64,
    ).eval()
    inputs = [
        torch.randn(
            *shape, device=device, dtype=torch.float64
        ).requires_grad_()
        for shape in [(2, 8, 3, 5), (2, 6, 3, 5), (2, 12), (2, 12)]
    ]
    residual, skip = block(*inputs)
    (residual.square().mean() + skip.square().mean()).backward()
    for value in (*inputs, *block.parameters()):
        assert value.grad is not None
        assert value.grad.isfinite().all()
    for value in inputs:
        assert value.grad is not None
        assert value.grad.abs().sum() > 0

    restored = ResidualBlock(
        channels=8,
        side_channels=6,
        embedding_channels=12,
        num_heads=2,
        device=device,
        dtype=torch.float64,
    ).eval()
    restored.load_state_dict(block.state_dict(), strict=True)
    with torch.inference_mode():
        actual = restored(*inputs)
        individual = [
            restored(*(value[i : i + 1] for value in inputs)) for i in range(2)
        ]
    for index, expected in enumerate((residual, skip)):
        torch.testing.assert_close(actual[index], expected)
        torch.testing.assert_close(
            torch.cat([output[index] for output in individual]), expected
        )
