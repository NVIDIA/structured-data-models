# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from sdm.nn import SinusoidalEmbedding
from sdm.testing import withCUDA


@withCUDA
def test_sinusoidal_embedding(device: torch.device) -> None:
    module = SinusoidalEmbedding()
    x = torch.ones(2, 3, 4, device=device)
    expected = x + x.new_tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)],
            [math.sin(2), math.cos(2), math.sin(0.02), math.cos(0.02)],
        ]
    )

    out = module(x)
    assert out.device == device
    torch.testing.assert_close(out, expected)
