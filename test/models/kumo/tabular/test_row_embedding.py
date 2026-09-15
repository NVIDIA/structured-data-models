# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.cache import Cache
from sdm.models.kumo.tabular.row_embedding import RowEmbedding
from sdm.testing import withCUDA


@withCUDA
def test_row_embedding(device: torch.device) -> None:
    encoder = RowEmbedding(
        num_classes=0,
        channels=16,
        num_layers=4,
        num_heads=2,
        group_size=3,
        num_frequencies=32,
        num_inducing_points=4,
        num_readout_tokens=2,
        device=device,
    )
    x = torch.randn(2, 5, 3, device=device)
    x[0, 0, 0] = torch.nan
    x[0, 4, 1] = torch.nan
    x[1, 1, 2] = torch.nan
    x[1, 3, 0] = torch.nan
    y = torch.randn(2, 3, device=device)
    categorical_mask = torch.zeros(2, 3, device=device, dtype=torch.bool)

    with torch.no_grad():
        expected = encoder(x, y, categorical_mask)
    assert expected.size() == (2, 5, 32)
    assert expected.device == device

    cache = Cache()
    with torch.no_grad():
        encoder(x[:, :3], y, categorical_mask, cache=cache)
        out = encoder(
            x[:, 3:],
            y[:, :0],
            categorical_mask,
            cache=cache.freeze(),
        )
    assert out.size() == (2, 2, 32)
    assert out.device == device
    torch.testing.assert_close(out, expected[:, 3:], atol=1e-5, rtol=1e-5)
