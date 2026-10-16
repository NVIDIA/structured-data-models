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
            context_size=32,
            prediction_size=6,
            patch_size=4,
            channels=32,
            hidden_channels=48,
            num_layers=2,
            num_heads=4,
            cross_channel_heads=4,
        ),
    )
    model = KumoForecast(cross_channel=cross_channel, pretrained=False)
    x = TableTensor.from_tensor(torch.randn(32, 2), columns=["a", "b"])
    y = TableTensor.from_tensor(torch.randn(32, 2), columns=["u", "v"])
    query = TableTensor.from_tensor(torch.empty(3, 0))

    out = model(x, y, query)
    assert out.shape == (3, 2)
    assert out.columns["numerical"] == ("u", "v")
    assert out.numerical.isfinite().all()

    model.fit(x, y)
    torch.testing.assert_close(model.predict(query).numerical, out.numerical)
