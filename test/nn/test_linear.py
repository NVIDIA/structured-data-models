# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.nn.linear import Linear
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    "shape", [(2, 0, 3), (2, 4, 0, 3), (0, 4, 3), (2, 4, 3)]
)
def test_linear(device: torch.device, shape: tuple[int, ...]) -> None:
    layer = Linear(3, 5, device=device)
    x = torch.randn(shape, device=device)
    out = torch.empty((*shape[:-1], 5), device=device)

    expected = layer(x)
    with torch.no_grad():
        actual = layer(x, out=out)

    assert actual is out
    torch.testing.assert_close(actual, expected)


@withCUDA
@pytest.mark.parametrize("shape", [(2, 4, 3), (2, 5, 4, 3)])
def test_linear_compile(device: torch.device, shape: tuple[int, ...]) -> None:
    layer = Linear(3, 5, device=device)
    x = torch.randn(shape, device=device)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)

    torch.testing.assert_close(compiled(x), layer(x))
    x = x.transpose(0, 1)  # Not contiguous.
    torch.testing.assert_close(compiled(x), layer(x))
