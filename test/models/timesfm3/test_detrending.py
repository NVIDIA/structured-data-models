# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import ColumnarTensor, TableTensor
from sdm.models.timesfm3 import TimesFM3
from sdm.models.timesfm3.recipe import TIME_COLUMN
from sdm.processing.execution import RecipeExecution
from sdm.testing import withCUDA


@withCUDA
def test_restores_each_target_trend_across_quantiles(
    device: torch.device,
) -> None:
    context = TableTensor(
        columns={"numerical": ["a", "b"], "id": [TIME_COLUMN]},
        numerical=torch.tensor(
            [[1.0, 10.0], [5.0, 20.0], [9.0, 30.0]], device=device
        ),
        id=ColumnarTensor((torch.arange(3, device=device),)),
    )
    execution = RecipeExecution(TimesFM3.default_recipe())
    execution.fit_transform(x=context, y=context, related_tables=None)
    quantiles = tuple(i / 10 for i in range(1, 10))
    quantile_offsets = torch.arange(len(quantiles), device=device) / 10
    residual = (
        torch.tensor([0.5, -0.5], device=device)[:, None, None]
        + quantile_offsets
    ).expand(2, 2, -1)
    prediction = TableTensor(
        columns={
            "numerical": [
                f"{name}__q{int(100 * q)}"
                for name in ("a", "b")
                for q in quantiles
            ],
            "id": [TIME_COLUMN],
        },
        numerical=residual.flatten(-2),
        id=ColumnarTensor((torch.arange(3, 5, device=device),)),
    )

    (restored,) = execution.inverse_transform_target((prediction,))
    expected = (
        torch.tensor([[13.5, 40.5], [16.5, 49.5]], device=device)[:, :, None]
        + quantile_offsets
    ).flatten(-2)
    torch.testing.assert_close(restored.numerical, expected)
    assert restored.columns == prediction.columns
