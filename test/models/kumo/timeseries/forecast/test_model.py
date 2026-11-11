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
    [(1, 120), (37, 600), (1200, 600)],
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


@withCUDA
@pytest.mark.parametrize("context_size", [1, 37, 512, 600])
@pytest.mark.parametrize("missing", ["none", "some", "all"])
def test_input_preparation_parity(
    monkeypatch: pytest.MonkeyPatch,
    context_size: int,
    missing: str,
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
    model = KumoForecast(pretrained=False, device=device)
    values = torch.arange(context_size * 4, device=device).float()
    values = values.reshape(context_size, 4).sin()
    if missing == "some":
        values[::7] = torch.nan
    elif missing == "all":
        values[:] = torch.nan

    x_context = TableTensor.from_tensor(values[:, :2], columns=["x1", "x2"])
    y_context = TableTensor.from_tensor(values[:, 2:], columns=["y1", "y2"])
    x_query = TableTensor.from_tensor(values.new_empty(120, 0))

    # Match the public SDK's raw zero imputation, with caller-side padding.
    reference = values.nan_to_num(nan=0)
    reference = torch.nn.functional.pad(
        reference,
        pad=(0, 0, max(512 - context_size, 0), 0),
        value=0,
    )
    with torch.inference_mode():
        actual = model(x_context, y_context, x_query)
        expected = model(
            TableTensor.from_tensor(reference[:, :2], columns=["x1", "x2"]),
            TableTensor.from_tensor(reference[:, 2:], columns=["y1", "y2"]),
            x_query,
        )

    torch.testing.assert_close(actual.numerical, expected.numerical)
    assert actual.numerical_columns == ("y1", "y2")
    assert actual.numerical.isfinite().all()
