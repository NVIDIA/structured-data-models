# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.models import Chronos2


@pytest.mark.parametrize("cross_learning", [False, True])
def test_forward(cross_learning: bool) -> None:
    model = Chronos2(pretrained=False)

    # Past and past-and-future covariates:
    x_context = TableTensor.from_tensor(
        torch.randn(2, 5, 4),
        columns=["x1", "x2", "x3", "x4"],
    )
    x_query = TableTensor.from_tensor(
        torch.randn(2, 3, 2),
        columns=["x2", "x4"],
    )

    # Target variates:
    y_context = TableTensor.from_tensor(
        torch.randn(2, 5, 3),
        columns=["y1", "y2", "y3"],
    )

    out = model(
        x_context=x_context,
        y_context=y_context,
        x_query=x_query,
        num_estimators=1,
        cross_learning=cross_learning,
    )
    assert out.size() == (2, 3, 3 * 21)
    assert out.numerical.isfinite().all()
    assert "y1__q10" in out.numerical_columns
    assert "y2__q50" in out.numerical_columns
    assert "y3__q90" in out.numerical_columns

    model.fit(
        x=x_context,
        y=y_context,
        num_estimators=1,
        context_only_columns=["x1", "x3"],
        cross_learning=cross_learning,
    )
    torch.testing.assert_close(model.predict(x_query).numerical, out.numerical)
