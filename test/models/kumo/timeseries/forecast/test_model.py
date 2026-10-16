# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import pytest
import torch

import sdm.processing as sp
from sdm import TableTensor
from sdm.models import KumoForecast
from sdm.models.kumo.timeseries.forecast import model as forecasting
from sdm.testing import withCUDA


@pytest.fixture(params=[False, True])
def model(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> KumoForecast:
    monkeypatch.setattr(
        forecasting,
        "_KumoForecast",
        partial(
            forecasting._KumoForecast,
            context_size=32,
            prediction_size=6,
            patch_size=4,
            channels=32,
            hidden_channels=48,
            num_layers=2,
            num_heads=4,
            head_channels=8,
            cross_channel_heads=4,
        ),
    )
    return KumoForecast(cross_channel=request.param, pretrained=False)


def test_forward_and_fit_predict(model: KumoForecast) -> None:
    x = TableTensor.from_tensor(torch.randn(32, 2), columns=["a", "b"])
    y = TableTensor.from_tensor(torch.randn(32, 2), columns=["u", "v"])
    query = TableTensor.from_tensor(
        torch.full((6, 2), float("nan")), columns=["a", "b"]
    )
    direct = model(x, y, query)
    model.fit(x, y)
    cached = model.predict(query)
    torch.testing.assert_close(cached.numerical, direct.numerical)
    assert direct.shape == (6, 2)
    assert direct.columns["numerical"] == ("u", "v")
    assert direct.numerical.isfinite().all()
    assert torch.is_inference(direct)
    torch.testing.assert_close(
        model.predict(query[:3]).numerical, direct.numerical[:3]
    )
    # Query values cannot affect a model that only consumes past covariates.
    changed = query.replace_blocks(numerical=torch.randn(6, 2))
    torch.testing.assert_close(
        model(x, y, changed).numerical, direct.numerical
    )


def test_univariate_without_covariates(model: KumoForecast) -> None:
    y = torch.randn(32, 1)
    direct = model(torch.empty(32, 0), y, torch.empty(3, 0))
    model.fit(torch.empty(32, 0), y)
    cached = model.predict(torch.empty(3, 0))
    assert direct.shape == (3, 1)
    torch.testing.assert_close(cached.numerical, direct.numerical)


def test_batch_dimensions(model: KumoForecast) -> None:
    x, y, query = (
        torch.randn(2, 32, 1),
        torch.randn(2, 32, 2),
        torch.empty(2, 4, 1),
    )
    actual = model(x, y, query, num_estimators=1).numerical
    expected = torch.stack(
        [model(x[i], y[i], query[i]).numerical for i in range(2)]
    )
    assert actual.shape == (2, 4, 2)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    model.fit(x, y, num_estimators=1)
    torch.testing.assert_close(model.predict(query).numerical, actual)


def test_uses_last_context_window(model: KumoForecast) -> None:
    x, y, query = torch.randn(40, 1), torch.randn(40, 1), torch.empty(3, 1)
    torch.testing.assert_close(
        model(x, y, query).numerical,
        model(x[-32:], y[-32:], query).numerical,
    )


def test_history_affects_forecast(model: KumoForecast) -> None:
    x, y, query = torch.randn(32, 2), torch.randn(32, 1), torch.empty(3, 2)
    prediction = model(x, y, query).numerical
    changed = model(x.flip(0), y, query).numerical
    if isinstance(model.model.cross_channel_attn, torch.nn.Identity):
        torch.testing.assert_close(changed, prediction)
    else:
        assert not torch.allclose(changed, prediction)
    assert not torch.allclose(model(x, y.flip(0), query).numerical, prediction)


def test_missing_observations(model: KumoForecast) -> None:
    x, y = torch.randn(32, 2), torch.randn(32, 1)
    x[0, 0], y[4, 0] = float("nan"), float("nan")
    assert model(x, y, torch.empty(3, 2)).numerical.isfinite().all()


def test_fully_missing_series(model: KumoForecast) -> None:
    """Keep forecasts finite with empty target and covariate histories."""
    x, y = torch.randn(2, 32, 2), torch.randn(2, 32, 2)
    x[0, :, 1] = float("nan")
    y[0, :, 0] = float("nan")
    y[1] = float("nan")
    query = torch.full((2, 3, 2), float("nan"))

    direct = model(x, y, query, num_estimators=1).numerical
    model.fit(x, y, num_estimators=1)
    cached = model.predict(query).numerical

    assert direct.shape == (2, 3, 2)
    assert direct.isfinite().all()
    torch.testing.assert_close(cached, direct)
    for i in range(2):
        torch.testing.assert_close(
            direct[i],
            model(x[i], y[i], query[i]).numerical,
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize("horizon", [0, 7])
def test_rejects_unsupported_horizon(
    model: KumoForecast, horizon: int
) -> None:
    with pytest.raises(ValueError, match="Number of query rows"):
        model(torch.empty(32, 0), torch.randn(32, 1), torch.empty(horizon, 0))
    model.fit(torch.empty(32, 0), torch.randn(32, 1))
    with pytest.raises(ValueError, match="Number of query rows"):
        model.predict(torch.empty(horizon, 0))


def test_rejects_short_history(model: KumoForecast) -> None:
    with pytest.raises(ValueError, match="at least 32 context rows"):
        model(torch.empty(31, 0), torch.randn(31, 1), torch.empty(3, 0))


def test_custom_recipe(model: KumoForecast) -> None:
    x, y, query = torch.randn(32, 1), torch.randn(32, 1), torch.empty(3, 1)
    recipe = sp.Recipe(target=sp.Standardize(), output=sp.AverageEstimators())
    direct = model(x, y, query, recipe=recipe)
    model.fit(x, y, recipe=recipe)
    torch.testing.assert_close(
        model.predict(query).numerical, direct.numerical
    )


def test_original_scale(model: KumoForecast) -> None:
    x, y, query = torch.empty(32, 0), torch.randn(32, 2), torch.empty(3, 0)
    prediction = model(x, y, query).numerical
    transformed = model(x, 3 * y + 7, query).numerical
    # RevIN adds epsilon to the standard deviation in standardized units.
    torch.testing.assert_close(
        transformed, 3 * prediction + 7, atol=2e-4, rtol=2e-4
    )


@withCUDA
def test_dtype_conversion(model: KumoForecast, device: torch.device) -> None:
    model.to(device)
    x = torch.empty(32, 0, device=device, dtype=torch.float64)
    y = torch.randn(32, 2, device=device, dtype=torch.float64)
    query = torch.empty(3, 0, device=device, dtype=torch.float64)
    prediction = model(x, y, query).numerical
    assert prediction.dtype == torch.float64
    assert prediction.device == device
    assert prediction.isfinite().all()


def test_meta_construction() -> None:
    model = KumoForecast(pretrained=False, device="meta")
    assert all(parameter.is_meta for parameter in model.parameters())
    with torch.inference_mode():
        out = model.model(torch.empty(2, 3, 512, device="meta"))
    assert out.shape == (2, 3, 72)
    assert out.is_meta
