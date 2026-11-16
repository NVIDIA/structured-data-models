# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.models.kumo.timeseries.forecast.missingness.backbone import (
    AdaptiveRevIN,
    CrossChannelAttention,
    MissingnessAdapter,
    ObservationAttentionBias,
)
from sdm.testing import withCUDA


@withCUDA
def test_observation_bias(device: torch.device) -> None:
    model = ObservationAttentionBias(2, device=device)
    observed = torch.tensor([[False, False, True, False, True]], device=device)
    gap = torch.tensor([[2, 2, 0, 1, 0]], device=device)
    pos = torch.arange(5, device=device)
    temporal = (
        -(pos[:, None] - pos[None, :]).abs()
        * torch.tensor(-2.0, device=device).exp()
    )
    confidence = observed * 0.3 - ~observed * 0.3 * gap / 32
    expected = (
        (temporal + confidence[:, None, :]).unsqueeze(1).expand(1, 2, 5, 5)
    )
    torch.testing.assert_close(model(observed), expected)


@withCUDA
def test_adaptive_statistics_exclude_padding(device: torch.device) -> None:
    model = AdaptiveRevIN(2, device=device)
    model.prior.fill_(2)
    model.initialized.fill_(True)
    x = torch.tensor(
        [[2.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], device=device
    )
    observed = torch.tensor(
        [[True, True, False, False], [False, False, False, False]],
        device=device,
    )
    valid = torch.tensor([True, False], device=device)
    mean, scale = model(x, observed, valid)
    torch.testing.assert_close(mean, x.new_tensor([[3.0], [0.0]]))
    torch.testing.assert_close(scale, x.new_full((2, 1), 1 + 1e-5))
    valid[:] = True  # A real all-missing auxiliary contributes to the rate.
    mean, scale = model(x, observed, valid)
    torch.testing.assert_close(mean, x.new_tensor([[1.5], [0.0]]))
    torch.testing.assert_close(scale, x.new_full((2, 1), 1.5 + 0.5e-5))
    assert (model.prior == 2).all()


@withCUDA
def test_cross_channel_matches_native_attention(device: torch.device) -> None:
    model = CrossChannelAttention(16, 4, 3, device=device)
    reference = torch.nn.MultiheadAttention(
        16, 4, batch_first=True, device=device
    )
    model.attn.qkv_lin.weight.data.copy_(reference.in_proj_weight)
    model.attn.qkv_lin.bias.data.copy_(reference.in_proj_bias)
    model.attn.out_lin.weight.data.copy_(reference.out_proj.weight)
    model.attn.out_lin.bias.data.copy_(reference.out_proj.bias)
    x = torch.randn(2, 4, 5, 16, device=device)
    observed = torch.ones(2, 4, 5, dtype=torch.bool, device=device)
    observed[:, :, 2] = False  # All-missing patch must still be finite.
    observed[:, -1] = False
    valid = torch.tensor([True, True, True, False], device=device)
    ids = torch.tensor([0, 1, 2, 2], device=device)
    hidden = x.transpose(1, 2).reshape(10, 4, 16) + model.channel_embed(ids)
    mask = ~observed.transpose(1, 2).reshape(10, 4)
    mask[mask.all(-1), 0] = False
    out, _ = reference(
        hidden, hidden, hidden, key_padding_mask=mask, need_weights=False
    )
    expected = (
        model.norm(hidden + model.residual_gate.sigmoid() * out)
        .reshape(2, 5, 4, 16)
        .transpose(1, 2)
    )
    expected[:, -1] = x[:, -1]
    torch.testing.assert_close(model(x, observed, valid), expected)


@withCUDA
def test_adapter_clean_and_padding_are_unchanged(device: torch.device) -> None:
    adapter = MissingnessAdapter(16, device=device)
    torch.nn.init.normal_(adapter.up.weight)
    x = torch.randn(2, 3, 5, 16, device=device)
    valid = torch.tensor([True, True, False], device=device)
    observed = torch.ones(2, 3, 5, dtype=torch.bool, device=device)
    observed[:, -1] = False
    torch.testing.assert_close(adapter(x, observed, valid), x, atol=0, rtol=0)
    observed[:, 0, 2:] = False
    result = adapter(x, observed, valid)
    torch.testing.assert_close(result[:, -1], x[:, -1], atol=0, rtol=0)
    assert not torch.allclose(result[:, :2, 2:], x[:, :2, 2:])
