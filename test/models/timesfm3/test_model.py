# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.models import TimesFM3


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
    assert out.numerical.isfinite().all()
    assert "y1__q10" in out.numerical_columns
    assert "y2__q50" in out.numerical_columns
    assert "y3__q90" in out.numerical_columns

    model.fit(
        x=x_context,
        y=y_context,
        context_only_columns=["x1", "x3"],
    )
    torch.testing.assert_close(model.predict(x_query).numerical, out.numerical)
