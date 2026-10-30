# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from functools import partial

import pytest
import torch

from sdm import TableTensor
from sdm.models import KumoForecast
from sdm.models.kumo.timeseries.forecast import model as forecasting
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("cross_channel", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    ("context_size", "horizon"),
    [(1, 120), (37, 120), (1024, 120), (37, 600), (1200, 600)],
)
def test_forward(
    monkeypatch: pytest.MonkeyPatch,
    cross_channel: bool,
    alpha: float,
    context_size: int,
    horizon: int,
    device: torch.device,
) -> None:
    monkeypatch.setattr(
        forecasting,
        "_KumoForecast",
        partial(
            forecasting._KumoForecast,
            channels=32,
            hidden_channels=48,
            num_layers=2,
            num_heads=4,
            cross_channel_heads=4,
        ),
    )
    model = KumoForecast(cross_channel, pretrained=False, device=device)

    # Past and past-and-future covariates:
    x_context = TableTensor.from_tensor(
        torch.randn(context_size, 2, device=device),
        columns=["x1", "x2"],
    )
    x_query = TableTensor.from_tensor(torch.randn(horizon, 0, device=device))

    # Target variates:
    y_context = TableTensor.from_tensor(
        torch.randn(context_size, 3, device=device),
        columns=["y1", "y2", "y3"],
    )

    with (
        nullcontext()
        if cross_channel or (alpha < 1 and context_size >= 512 + horizon)
        else pytest.warns(RuntimeWarning, match="Historical covariates")
    ):
        out = model(x_context, y_context, x_query, alpha=alpha)
    assert out.size() == (horizon, 3)
    assert out.device == device
    assert out.numerical.isfinite().all()
    assert "y1" in out.numerical_columns
    assert "y2" in out.numerical_columns
    assert "y3" in out.numerical_columns

    if context_size < 512:
        padded_x_context = TableTensor.from_tensor(
            torch.nn.functional.pad(
                x_context.numerical,
                pad=(0, 0, 512 - context_size, 0),
                value=float("nan"),
            ),
            columns=x_context.numerical_columns,
        )
        padded_y_context = TableTensor.from_tensor(
            torch.nn.functional.pad(
                y_context.numerical,
                pad=(0, 0, 512 - context_size, 0),
                value=float("nan"),
            ),
            columns=y_context.numerical_columns,
        )
        with (
            nullcontext()
            if cross_channel
            else pytest.warns(RuntimeWarning, match="Historical covariates")
        ):
            expected = model(
                padded_x_context,
                padded_y_context,
                x_query,
                alpha=alpha,
            )
        torch.testing.assert_close(out.numerical, expected.numerical)
