# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.models.timesfm3.normalization import PerDimScale
from sdm.testing import withCUDA


@withCUDA
def test_per_dim_scale(device: torch.device) -> None:
    num_dims = 8
    module = PerDimScale(num_dims=num_dims, device=device)
    with torch.no_grad():
        module.per_dim_scale.fill_(1.0)
    tensor = torch.ones(2, 3, num_dims, device=device)

    out = module(tensor)

    torch.testing.assert_close(
        out,
        torch.full_like(tensor, 0.669855025622358),
    )


@withCUDA
def test_per_dim_scale_dtype_device(device: torch.device) -> None:
    dtype = torch.float64
    module = PerDimScale(num_dims=4, device=device, dtype=dtype)
    tensor = torch.ones(2, 4, device=device, dtype=dtype)

    out = module(tensor)

    assert out.dtype == dtype
    assert out.device == device
    assert module.per_dim_scale.dtype == dtype
    assert module.per_dim_scale.device == device
    assert tuple(module.state_dict()) == ("per_dim_scale",)
