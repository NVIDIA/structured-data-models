# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from sdm.nn import SoftplusScale
from sdm.testing import withCUDA


@withCUDA
def test_per_dim_scale(device: torch.device) -> None:
    num_dims = 8
    module = SoftplusScale(
        channels=num_dims,
        multiplier=1.442695041 / math.sqrt(num_dims),
        device=device,
    )
    with torch.no_grad():
        module.weight.fill_(1.0)
    tensor = torch.ones(2, 3, num_dims, device=device)

    out = module(tensor)

    torch.testing.assert_close(
        out,
        torch.full_like(tensor, 0.669855025622358),
    )


@withCUDA
def test_per_dim_scale_dtype_device(device: torch.device) -> None:
    dtype = torch.float64
    module = SoftplusScale(
        channels=4,
        multiplier=1.442695041 / math.sqrt(4),
        device=device,
        dtype=dtype,
    )
    tensor = torch.ones(2, 4, device=device, dtype=dtype)

    out = module(tensor)

    assert out.dtype == dtype
    assert out.device == device
    assert module.weight.dtype == dtype
    assert module.weight.device == device
    assert tuple(module.state_dict()) == ("weight",)
