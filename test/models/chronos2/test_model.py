# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.models import Chronos2


def test_forward() -> None:
    model = Chronos2(pretrained=False)

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
    assert out.size() == (3, 3 * 21)
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


@pytest.mark.parametrize("estimator_batch_size", [1, None])
def test_cross_learning(estimator_batch_size: int | None) -> None:
    model = Chronos2(pretrained=False)
    data = torch.randn(2, 2, 5, 2)  # [E, B, R, Y]
    context = TableTensor.from_tensor(data)
    features = TableTensor(size=(2, 2, 5))
    query = TableTensor(size=(2, 2, 3))

    out = model(
        features,
        context,
        query,
        cross_learning=True,
        estimator_batch_size=estimator_batch_size,
    )

    # Group examples into target columns separately for each estimator.
    grouped = TableTensor.from_tensor(data.transpose(-3, -2).flatten(-2))
    expected = model(
        TableTensor(size=(2, 5)),
        grouped,
        TableTensor(size=(2, 3)),
        estimator_batch_size=estimator_batch_size,
    ).numerical
    expected = expected.reshape(3, 2, 2 * 21).transpose(0, 1)
    assert out.size() == (2, 3, 2 * 21)
    torch.testing.assert_close(out.numerical, expected)

    model.fit(
        features,
        context,
        cross_learning=True,
        estimator_batch_size=estimator_batch_size,
    )
    torch.testing.assert_close(model.predict(query).numerical, out.numerical)
