# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from typing import Any

import pytest
import torch

from sdm import TableTensor
from sdm.models import KumoForecast
from sdm.models.kumo.timeseries.forecast import model as kumo_forecast
from sdm.processing import InvertibleMixin
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

    class _KumoForecast(kumo_forecast._KumoForecast):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(
                channels=32,
                hidden_channels=48,
                num_layers=2,
                num_heads=4,
                cross_channel_heads=4,
                **kwargs,
            )

    monkeypatch.setattr(kumo_forecast, "_KumoForecast", _KumoForecast)
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


def test_missingness_default_recipe() -> None:
    model = KumoForecast(missingness=True, pretrained=False)
    table = TableTensor.from_tensor(
        torch.tensor([[1.0, 5.0], [3.0, 5.0], [torch.nan, torch.nan]])
    )
    expected = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [torch.nan, torch.nan]])
    recipe = model.default_recipe()
    assert isinstance(recipe.target, InvertibleMixin)

    features = recipe.features.fit_transform(table)
    target = recipe.target.fit_transform(table)
    torch.testing.assert_close(features.numerical, expected, equal_nan=True)
    torch.testing.assert_close(target.numerical, expected, equal_nan=True)
    restored = recipe.target.inverse_transform(target)
    torch.testing.assert_close(
        restored.numerical, table.numerical, equal_nan=True
    )

    other = model.default_recipe().target.fit_transform(
        TableTensor.from_tensor(table.numerical + 10)
    )
    torch.testing.assert_close(other.numerical, expected, equal_nan=True)
    restored = recipe.target.inverse_transform(target)
    torch.testing.assert_close(
        restored.numerical, table.numerical, equal_nan=True
    )
