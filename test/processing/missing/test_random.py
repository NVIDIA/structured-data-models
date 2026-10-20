# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.processing import MaskRandom


def test_masks_cells_at_the_configured_rate() -> None:
    rate = 0.05
    table = TableTensor(
        numerical=torch.randn(4, 4000, 3),
        categorical=CategoricalTensor(
            code=torch.randint(3, (4, 4000, 2)),
            categories=(torch.arange(3),) * 2,
        ),
    )

    output = MaskRandom(rate=(rate, rate)).fit_transform(table)

    numerical_missing = output.numerical.isnan()
    categorical_missing = output.categorical.code == -1
    missing = torch.cat((numerical_missing, categorical_missing), dim=-1)
    actual_rate = missing.float().mean(dim=(-2, -1))
    torch.testing.assert_close(
        actual_rate,
        torch.full_like(actual_rate, rate),
        rtol=0.0,
        atol=0.01,
    )
    assert output.numerical[~numerical_missing].equal(
        table.numerical[~numerical_missing]
    )
    assert output.categorical.code[~categorical_missing].equal(
        table.categorical.code[~categorical_missing]
    )
    assert all(
        actual.equal(expected)
        for actual, expected in zip(
            output.categorical.categories,
            table.categorical.categories,
            strict=True,
        )
    )
