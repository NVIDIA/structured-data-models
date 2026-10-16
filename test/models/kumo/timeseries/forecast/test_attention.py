# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.models.kumo.timeseries.forecast.attention import CrossChannelAttention
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_cross_channel_attention(
    device: torch.device,
    batch_shape: tuple[int, ...],
) -> None:
    reference = torch.nn.MultiheadAttention(
        embed_dim=8, num_heads=2, batch_first=True, device=device
    ).eval()
    norm = torch.nn.LayerNorm(8, device=device)
    model = CrossChannelAttention(
        channels=8, num_heads=2, device=device
    ).eval()
    with torch.no_grad():
        model.attn.qkv_lin.weight.copy_(reference.in_proj_weight)
        model.attn.qkv_lin.bias.copy_(reference.in_proj_bias)
        model.attn.out_lin.load_state_dict(reference.out_proj.state_dict())
        model.norm.load_state_dict(norm.state_dict())
    x = torch.randn(*batch_shape, 3, 4, 8, device=device)
    flattened = x.transpose(-3, -2).reshape(-1, 3, 8)
    with torch.inference_mode():
        attended, _ = reference(flattened, flattened, flattened)
        expected = norm(flattened + attended)
        expected = expected.reshape(*batch_shape, 4, 3, 8).transpose(-3, -2)
        actual = model(x)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        permutation = torch.tensor([2, 0, 1], device=device)
        torch.testing.assert_close(
            model(x[..., permutation, :, :]),
            actual[..., permutation, :, :],
            atol=1e-6,
            rtol=1e-5,
        )
        changed = x.clone()
        changed[..., 0, :] += 10
        torch.testing.assert_close(
            model(changed)[..., 1:, :], actual[..., 1:, :]
        )
