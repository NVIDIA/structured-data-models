# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.nn import MonotoneQuantiles
from sdm.testing import withCUDA


@withCUDA
def test_monotone_quantiles(device: torch.device) -> None:
    levels = [0.01, 0.1, 0.5, 0.8, 0.99]
    head = MonotoneQuantiles(levels, width_factor=3).to(device)
    x = torch.randn(2, 7, 5, device=device)
    out = head(x)
    torch.testing.assert_close(out[..., 2], x[..., 2])
    assert (out[..., 1:] >= out[..., :-1]).all()
    expected_gap = (
        3 * (levels[3] - levels[2]) * torch.nn.functional.softplus(x[..., 3])
    )
    torch.testing.assert_close(out[..., 3] - out[..., 2], expected_gap)
