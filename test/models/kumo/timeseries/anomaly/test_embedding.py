# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from sdm.models.kumo.timeseries.anomaly.embedding import DiffusionEmbedding
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("out_channels", [None, 12])
def test_diffusion_embedding(
    device: torch.device,
    dtype: torch.dtype,
    out_channels: int | None,
) -> None:
    module = DiffusionEmbedding(
        num_steps=1000,
        channels=128,
        out_channels=out_channels,
        device=device,
        dtype=dtype,
    )
    step = torch.tensor([0, 3, 999], device=device)
    # The released table concatenates sine then cosine, with frequencies
    # increasing from 1 to 10,000 rather than a standard positional encoding.
    frequencies = 10.0 ** (torch.arange(64, dtype=torch.float32) / 63 * 4)
    angles = (
        torch.tensor([0, 3, 999], dtype=torch.float32)[:, None] * frequencies
    )
    table = torch.cat([angles.sin(), angles.cos()], dim=-1).to(
        device=device, dtype=dtype
    )
    expected = F.silu(
        F.linear(
            input=table,
            weight=module.projection1.weight,
            bias=module.projection1.bias,
        )
    )
    expected = F.silu(
        F.linear(
            input=expected,
            weight=module.projection2.weight,
            bias=module.projection2.bias,
        )
    )
    actual = module(step)
    assert actual.shape == (3, out_channels or 128)
    assert actual.dtype == dtype
    assert actual.device == device
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(module(step[:1]), actual[:1])
    assert "embedding" not in module.state_dict()


@withCUDA
def test_embedding_dtype_conversion(device: torch.device) -> None:
    module = DiffusionEmbedding(num_steps=10, channels=8).to(
        device=device, dtype=torch.float64
    )
    result = module(torch.tensor([2, 8], device=device))
    assert result.dtype == torch.float64
    assert result.isfinite().all()


@pytest.mark.parametrize("channels", [1, 2, 3, 7])
def test_invalid_embedding_width(channels: int) -> None:
    with pytest.raises(ValueError, match="even and at least four"):
        DiffusionEmbedding(num_steps=10, channels=channels)
