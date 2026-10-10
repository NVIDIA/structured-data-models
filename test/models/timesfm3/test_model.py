# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.models.timesfm3 import TimesFM3
from sdm.models.timesfm3.model import _TimesFM3
from sdm.testing import withCUDA


def test_forward() -> None:
    model = TimesFM3(pretrained=False)

    # Past and past-and-future covariates:
    x_context = TableTensor.from_tensor(
        torch.randn(5, 4),
        columns=["x1", "x2", "x3", "x4"],
    )
    x_query = TableTensor.from_tensor(
        torch.randn(3, 2),
        columns=["x2", "x4"],
    )

    # Target variates:
    y_context = TableTensor.from_tensor(
        torch.randn(5, 3),
        columns=["y1", "y2", "y3"],
    )

    out = model(x_context, y_context, x_query)
    assert out.size() == (3, 3 * 9)
    assert "y1__q10" in out.columns["numerical"]
    assert "y2__q50" in out.columns["numerical"]
    assert "y3__q90" in out.columns["numerical"]

    model.fit(x_context, y_context)
    out = model.predict(x_query)
    assert out.allclose(model.predict(x_query))


@withCUDA
def test_core(
    device: torch.device,
) -> None:
    model = _TimesFM3(
        input_patch_size=2,
        output_patch_size=4,
        channels=8,
        num_layers=2,
        num_heads=2,
        device=device,
    )

    out = model(
        x_context=torch.randn(5, 3, device=device),
        x_query=torch.randn(8, 3, device=device),
        x_context_only=torch.randn(5, 1, device=device),
        y=torch.randn(5, 2, device=device),
    )
    assert out.device == device
    assert out.size() == (8, 2, 9)
