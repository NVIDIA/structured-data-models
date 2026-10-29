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
def test_forward(
    monkeypatch: pytest.MonkeyPatch,
    cross_channel: bool,
    alpha: float,
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
        torch.randn(1024, 2, device=device),
        columns=["x1", "x2"],
    )
    x_query = TableTensor.from_tensor(torch.randn(120, 0, device=device))

    # Target variates:
    y_context = TableTensor.from_tensor(
        torch.randn(1024, 3, device=device),
        columns=["y1", "y2", "y3"],
    )

    with (
        nullcontext()
        if cross_channel or alpha < 1
        else pytest.warns(RuntimeWarning, match="Historical covariates")
    ):
        out = model(x_context, y_context, x_query, alpha=alpha)
    assert out.size() == (120, 3)
    assert out.device == device
    assert out.numerical.isfinite().all()
    assert "y1" in out.numerical_columns
    assert "y2" in out.numerical_columns
    assert "y3" in out.numerical_columns
