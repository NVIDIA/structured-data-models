# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.nn import RevIN
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("affine", [False, True])
def test_revin(device: torch.device, affine: bool) -> None:
    norm = RevIN(channels=2, affine=affine, device=device)
    x = torch.randn(2, 3, 2, 8, device=device)
    mean, std = norm.compute_stats(x)
    torch.testing.assert_close(mean, x.mean(dim=-1, keepdim=True))
    torch.testing.assert_close(std, x.std(dim=-1, correction=0, keepdim=True))

    out = norm(x, mean, std)
    expected = (x - mean) / (std + norm.eps)
    torch.testing.assert_close(out, expected)

    torch.testing.assert_close(norm.inverse(out, mean, std), x)
