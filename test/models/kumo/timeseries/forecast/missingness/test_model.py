# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import pytest
import torch

from sdm import TableTensor
from sdm.models import KumoForecastMissingness
from sdm.models.kumo.timeseries.forecast.missingness import (
    model as forecasting,
)
from sdm.testing import withCUDA


@pytest.fixture
def model(
    monkeypatch: pytest.MonkeyPatch, device: torch.device
) -> KumoForecastMissingness:
    monkeypatch.setattr(
        forecasting,
        "MissingnessBackbone",
        partial(
            forecasting.MissingnessBackbone,
            channels=32,
            hidden_channels=48,
            num_layers=2,
            num_heads=4,
            cross_channel_heads=4,
        ),
    )
    return KumoForecastMissingness(pretrained=False, device=device)


@withCUDA
@pytest.mark.parametrize("horizon", [1, 24, 25, 120, 512])
@pytest.mark.parametrize("channels", [1, 7, 9])
def test_tensor_forecast(
    model: KumoForecastMissingness,
    device: torch.device,
    horizon: int,
    channels: int,
) -> None:
    history = torch.randn(180, channels, device=device)
    history[::3] = torch.nan
    history[70:100, -1] = torch.inf
    original = history.clone()
    out = model(history[:, 1:], history[:, :1], history.new_empty(horizon, 0))
    assert out.shape == (horizon, 1)
    assert out.device == device
    assert out.numerical.isfinite().all()
    torch.testing.assert_close(history, original, equal_nan=True)
    short = model(history[:, 1:], history[:, :1], history.new_empty(1, 0))
    torch.testing.assert_close(out.numerical[:1], short.numerical)


@withCUDA
def test_named_alignment_and_multiple_targets(
    model: KumoForecastMissingness, device: torch.device
) -> None:
    history = torch.randn(180, 4, device=device)
    history[::2, 0] = torch.nan
    history[:, 2] = torch.nan
    target = TableTensor.from_tensor(history[:, :1], columns=["target"])
    features = TableTensor.from_tensor(
        history[:, 1:], columns=["MULL", "HULL", "HUFL"]
    )
    query = TableTensor.from_tensor(history.new_empty(49, 0))
    out = model(features, target, query)
    reordered = model(features[..., ["HUFL", "MULL", "HULL"]], target, query)
    torch.testing.assert_close(out.numerical, reordered.numerical)
    all_targets = TableTensor.from_tensor(
        history, columns=["target", "MULL", "HULL", "HUFL"]
    )
    all_out = model(history.new_empty(180, 0), all_targets, query)
    assert all_out.numerical_columns == all_targets.numerical_columns
    torch.testing.assert_close(out.numerical, all_out.numerical[:, :1])
    assert all_out.numerical.isfinite().all()


@withCUDA
def test_estimator_batching(
    model: KumoForecastMissingness, device: torch.device
) -> None:
    y = torch.randn(3, 160, 2, device=device)
    y[0, 20:, 1] = torch.nan
    y[1, 30:130] = torch.nan
    x = y.new_empty(3, 160, 0)
    query = y.new_empty(3, 49, 0)
    batched = model(x, y, query, estimator_batch_size=None)
    separate = model(x, y, query, estimator_batch_size=1)
    torch.testing.assert_close(batched.numerical, separate.numerical)


def test_history_scope(model: KumoForecastMissingness) -> None:
    y = torch.randn(240, 1)
    y[-90:-10] = torch.nan
    y[:120] += 10
    query = torch.empty(48, 0)
    scoped = model(torch.empty(240, 0), y, query, history_rows=120)
    sliced = model(torch.empty(120, 0), y[-120:], query)
    full = model(torch.empty(240, 0), y, query)
    torch.testing.assert_close(scoped.numerical, sliced.numerical)
    assert not torch.allclose(full.numerical, scoped.numerical)


def test_missing_is_not_zero(model: KumoForecastMissingness) -> None:
    y = torch.linspace(-1, 1, 150).unsqueeze(-1)
    missing = y.clone()
    missing[-80:-16] = torch.nan
    filled = missing.nan_to_num()
    query = torch.empty(24, 0)
    a = model(torch.empty(150, 0), missing, query)
    b = model(torch.empty(150, 0), filled, query)
    assert not torch.allclose(a.numerical, b.numerical)


@pytest.mark.parametrize(
    ("rows", "horizon", "query_cols", "message"),
    [
        (95, 24, 0, "96 context"),
        (96, 0, 0, "horizon"),
        (96, 513, 0, "horizon"),
        (96, 24, 1, "zero columns"),
    ],
)
def test_invalid_context(
    model: KumoForecastMissingness,
    rows: int,
    horizon: int,
    query_cols: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        model(
            torch.empty(rows, 0),
            torch.randn(rows, 1),
            torch.empty(horizon, query_cols),
        )


def test_missing_target_and_unknown_channels(
    model: KumoForecastMissingness,
) -> None:
    with pytest.raises(ValueError, match="observed values"):
        model(
            torch.empty(96, 0),
            torch.full((96, 1), torch.nan),
            torch.empty(24, 0),
        )
    x = TableTensor.from_tensor(torch.randn(96, 1), columns=["sensor"])
    y = TableTensor.from_tensor(torch.randn(96, 1), columns=["target"])
    with pytest.raises(ValueError, match="positional"):
        model(x, y, torch.empty(24, 0))
    out = model(x, y, torch.empty(24, 0), channel_alignment="positional")
    assert out.numerical.isfinite().all()
    with pytest.raises(TypeError, match="alpha"):
        model(x, y, torch.empty(24, 0), alpha=0.5)


def test_dtype_and_state_roundtrip(
    model: KumoForecastMissingness,
) -> None:
    clone = KumoForecastMissingness(pretrained=False).double()
    clone.load_state_dict(model.state_dict())
    model.double()
    y = torch.randn(120, 2, dtype=torch.float64)
    y[::4] = torch.nan
    args = (
        torch.empty(120, 0, dtype=torch.float64),
        y,
        torch.empty(25, 0, dtype=torch.float64),
    )
    a, b = model(*args), clone(*args)
    assert a.numerical.dtype == torch.float64
    torch.testing.assert_close(a.numerical, b.numerical)
