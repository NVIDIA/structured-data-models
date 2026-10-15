# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch import Tensor

from sdm.models.kumo.timeseries.anomaly.block import (
    KumoAnomalyTransformerBlock,
    ResidualBlock,
)
from sdm.testing import withCUDA


def reference_attention(
    layer: KumoAnomalyTransformerBlock, x: Tensor
) -> Tensor:
    """Evaluate upstream's sequence-first PyTorch encoder layer."""
    state = layer.state_dict()
    reference = torch.nn.TransformerEncoderLayer(
        d_model=x.size(-1),
        nhead=layer.attn.num_query_heads,
        dim_feedforward=state["mlp.0.weight"].size(0),
        dropout=0.0,
        activation="gelu",
        device=x.device,
        dtype=x.dtype,
    )
    names = {
        "self_attn.in_proj_": "attn.qkv_lin.",
        "self_attn.out_proj.": "attn.out_lin.",
        "linear1.": "mlp.0.",
        "linear2.": "mlp.3.",
        "norm1.": "attn_residual_norm.",
        "norm2.": "mlp_residual_norm.",
    }
    reference.load_state_dict(
        {
            target + key: state[source + key]
            for target, source in names.items()
            for key in ("weight", "bias")
        }
    )
    return reference.eval()(x.transpose(0, 1)).transpose(0, 1)


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
    ("features", "length", "diffusion_batch", "strategy_batch", "training"),
    [
        (3, 5, 1, 2, False),
        (1, 5, 1, 2, False),
        (3, 1, 1, 2, False),
        (1, 1, 1, 2, False),
        (3, 5, 2, 1, False),
        (3, 5, 1, 2, True),
    ],
)
def test_residual_block(
    device: torch.device,
    features: int,
    length: int,
    diffusion_batch: int,
    strategy_batch: int,
    training: bool,
) -> None:
    block = ResidualBlock(
        channels=8,
        side_channels=6,
        embedding_channels=12,
        num_heads=2,
        dropout=0.0,
        device=device,
    ).train(training)
    x = torch.randn(2, 8, features, length, device=device)
    side = torch.randn(2, 6, features, length, device=device)
    diffusion = torch.randn(diffusion_batch, 12, device=device)
    strategy = torch.randn(strategy_batch, 12, device=device)
    with torch.inference_mode():
        actual = block(x, side, diffusion, strategy)
        expected = reference_block(block, x, side, diffusion, strategy)
    for result, reference in zip(actual, expected, strict=True):
        assert result.shape == x.shape
        assert result.dtype == x.dtype
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
    assert residual.dtype == skip.dtype == torch.float64
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
