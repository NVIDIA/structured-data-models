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
@pytest.mark.parametrize(
    ("dtype", "autocast_dtype"),
    [
        (torch.float32, None),
        (torch.float64, None),
        (torch.float16, None),
        (torch.bfloat16, None),
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
    ],
)
def test_forward(
    monkeypatch: pytest.MonkeyPatch,
    cross_channel: bool,
    device: torch.device,
    dtype: torch.dtype,
    autocast_dtype: torch.dtype | None,
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
    model.to(dtype)
    stream_dtypes = []
    norm_dtypes = []
    model.model.icl.norm.register_forward_hook(
        lambda module, args, out: norm_dtypes.append(out.dtype)
    )
    model.model.icl.register_forward_pre_hook(
        lambda module, args: stream_dtypes.append(args[0].dtype)
    )
    with (
        torch.inference_mode(),
        torch.autocast(
            device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ),
    ):
        # Past and past-and-future covariates:
        x_context = TableTensor.from_tensor(
            torch.randn(512, 2, device=device, dtype=dtype),
            columns=["x1", "x2"],
        )
        x_query = TableTensor.from_tensor(torch.randn(3, 0, device=device))

        # Target variates:
        y_context = TableTensor.from_tensor(
            torch.randn(512, 3, device=device, dtype=dtype),
            columns=["y1", "y2", "y3"],
        )

        with (
            nullcontext()
            if cross_channel
            else pytest.warns(RuntimeWarning, match="Historical covariates")
        ):
            out = model(x_context, y_context, x_query)
        assert out.size() == (3, 3)
        assert out.device == device
        assert out.numerical.isfinite().all()
        assert "y1" in out.numerical_columns
        assert "y2" in out.numerical_columns
        assert "y3" in out.numerical_columns

        with (
            nullcontext()
            if cross_channel
            else pytest.warns(RuntimeWarning, match="Historical covariates")
        ):
            model.fit(
                x=x_context,
                y=y_context,
                context_only_columns=["x1", "x2"],
            )
        torch.testing.assert_close(
            model.predict(x_query).numerical, out.numerical
        )

        empty = torch.empty(512, 0, device=device, dtype=dtype)
        out = model(
            empty, empty, torch.empty(3, 0, device=device, dtype=dtype)
        )
        assert out.size() == (1, 3, 0)
    assert stream_dtypes
    assert all(value == dtype for value in stream_dtypes)
    assert norm_dtypes
    assert all(value == dtype for value in norm_dtypes)
