# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import pytest
import torch

from sdm import TableTensor
from sdm.models import KumoForecast
from sdm.models.kumo.timeseries.forecast import model as forecasting


@pytest.mark.parametrize("cross_channel", [False, True])
def test_forward(monkeypatch: pytest.MonkeyPatch, cross_channel: bool) -> None:
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
    model = KumoForecast(cross_channel, pretrained=False)

    # Past and past-and-future covariates:
    x_context = TableTensor.from_tensor(
        torch.randn(512, 2),
        columns=["x1", "x2"],
    )
    x_query = TableTensor.from_tensor(torch.randn(3, 0))

    # Target variates:
    y_context = TableTensor.from_tensor(
        torch.randn(512, 3),
        columns=["y1", "y2", "y3"],
    )

    out = model(x_context, y_context, x_query)
    assert out.size() == (3, 3)
    assert out.numerical.isfinite().all()
    assert "y1" in out.columns["numerical"]
    assert "y2" in out.columns["numerical"]
    assert "y3" in out.columns["numerical"]

    model.fit(
        x=x_context,
        y=y_context,
        context_only_columns=["x1", "x2"],
    )
    torch.testing.assert_close(model.predict(x_query).numerical, out.numerical)
