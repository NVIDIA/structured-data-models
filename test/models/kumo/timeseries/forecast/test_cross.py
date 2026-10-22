# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from sdm.models.kumo.timeseries.forecast.cross import CrossChannelAttention
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_cross_channel_attention(
    batch_shape: tuple[int, ...],
    device: torch.device,
) -> None:
    model = CrossChannelAttention(
        channels=8,
        num_heads=2,
        device=device,
    )

    x = torch.randn(*batch_shape, 3, 4, 8, device=device)

    with torch.no_grad():
        out = model(x)
        chunked = model(x, batch_size_limit=1)
        torch.testing.assert_close(chunked, out)

    assert out.size() == (*batch_shape, 3, 4, 8)
    assert out.dtype == x.dtype
    assert out.device == device

    x.requires_grad_()
    model(x).square().mean().backward()
    assert x.grad is not None
    assert x.grad.isfinite().all()
    assert all(
        p.grad is not None and p.grad.isfinite().all()
        for p in model.parameters()
    )
