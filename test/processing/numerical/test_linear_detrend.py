# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import ColumnarTensor, TableTensor
from sdm.processing import LinearDetrend
from sdm.testing import withCUDA


def _table(
    values: list[float], steps: list[int], device: torch.device
) -> TableTensor:
    return TableTensor(
        columns={"numerical": ["value"], "id": ["step"]},
        numerical=torch.tensor(values, device=device)[:, None],
        id=ColumnarTensor((torch.tensor(steps, device=device),)),
    )


@withCUDA
def test_linear_detrend_uses_steps_on_context_and_future(
    device: torch.device,
) -> None:
    context = _table([1.0, 5.0, 9.0], [10, 12, 14], device)
    future = _table([20.0, 22.0], [18, 22], device)
    forecast = _table([0.5, -0.5], [18, 22], device)
    detrender = LinearDetrend("step").fit(context)

    torch.testing.assert_close(
        detrender.transform(context).numerical,
        torch.zeros_like(context.numerical),
    )
    torch.testing.assert_close(
        detrender.transform(future).numerical,
        future.numerical.new_tensor([[3.0], [-3.0]]),
    )
    torch.testing.assert_close(
        detrender.inverse_transform(forecast).numerical,
        forecast.numerical.new_tensor([[17.5], [24.5]]),
    )


@withCUDA
def test_linear_detrend_threshold_skips_unhelpful_trend(
    device: torch.device,
) -> None:
    context = _table([0.0, 10.0, 0.0, 10.0], [0, 1, 2, 3], device)
    future = _table([1.0, 2.0], [4, 5], device)
    detrender = LinearDetrend("step", threshold=0.5)

    torch.testing.assert_close(
        detrender.fit_transform(context).numerical, context.numerical
    )
    torch.testing.assert_close(
        detrender.transform(future).numerical, future.numerical
    )
    torch.testing.assert_close(
        detrender.inverse_transform(future).numerical, future.numerical
    )


@withCUDA
def test_linear_detrend_ignores_missing_context_values(
    device: torch.device,
) -> None:
    context = _table([1.0, float("nan"), 9.0], [0, 1, 2], device)
    future = _table([13.0], [3], device)
    detrender = LinearDetrend("step", threshold=0.5).fit(context)

    transformed = detrender.transform(context).numerical
    torch.testing.assert_close(
        transformed[[0, 2]], torch.zeros_like(transformed[[0, 2]])
    )
    assert transformed[1].isnan().all()
    torch.testing.assert_close(
        detrender.transform(future).numerical,
        torch.zeros_like(future.numerical),
    )


@withCUDA
def test_linear_detrend_threshold_handles_large_offsets(
    device: torch.device,
) -> None:
    step = torch.arange(1024, device=device)
    time = step.float()
    trend = 100_000 + 0.01 * time
    wave = (0.1 * time).sin()
    values = torch.stack((trend + wave, trend + 4 * wave), dim=-1)
    table = TableTensor(
        columns={"numerical": ["low_noise", "high_noise"], "id": ["step"]},
        numerical=values,
        id=ColumnarTensor((step,)),
    )

    transformed = LinearDetrend("step", threshold=0.5).fit_transform(table)

    assert transformed.numerical[:, 0].abs().max() < 2
    torch.testing.assert_close(transformed.numerical[:, 1], values[:, 1])
