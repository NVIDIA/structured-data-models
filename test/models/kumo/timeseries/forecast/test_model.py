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
@pytest.mark.parametrize(
    ("context_size", "horizon"),
    [(1, 120), (37, 600), (511, 120), (512, 120), (513, 600), (1200, 600)],
)
@pytest.mark.parametrize("cross_channel", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_forward(
    monkeypatch: pytest.MonkeyPatch,
    context_size: int,
    horizon: int,
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
        torch.randn(context_size, 2, device=device),
        columns=["x1", "x2"],
    )
    x_query = TableTensor.from_tensor(torch.randn(horizon, 0, device=device))

    # Target variates:
    y_context = TableTensor.from_tensor(
        torch.randn(context_size, 3, device=device),
        columns=["y1", "y2", "y3"],
    )

    if context_size < 512:
        with pytest.raises(
            ValueError,
            match="Expected at least 512 context rows",
        ):
            model(x_context, y_context, x_query, alpha=alpha)
        return

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


def test_missingness_placeholder() -> None:
    model = KumoForecast(missingness=True, pretrained=False)
    with pytest.raises(
        NotImplementedError, match="Missingness-aware forecasting"
    ):
        model(torch.empty(96, 0), torch.randn(96, 1), torch.empty(24, 0))

    with pytest.raises(NotImplementedError, match="checkpoint loading"):
        KumoForecast(missingness=True)


def test_missingness_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    def forecast(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == 96
        assert x.isnan().any()
        return x[..., -1:] + torch.arange(1, 25, device=x.device)

    monkeypatch.setattr(
        forecasting._KumoForecastMissingness, "forward", forecast
    )
    model = KumoForecast(missingness=True, pretrained=False)
    history = torch.zeros(96, 2)
    history[-2] = torch.nan
    out = model(history[:, 1:], history[:, :1], torch.empty(49, 0))
    torch.testing.assert_close(
        out.numerical, torch.arange(1, 50, dtype=torch.float32).unsqueeze(-1)
    )
