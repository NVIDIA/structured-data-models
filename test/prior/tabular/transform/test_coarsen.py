# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.prior import LogUniform, LogUniformInt
from sdm.prior.tabular.transform import Coarsen


def test_coarsen_reduces_and_standardizes_numerical_category_values() -> None:
    numerical = torch.randn(500, 4, dtype=torch.float64) * 3 + 2
    numerical[0, -1] = torch.nan
    categories = (torch.arange(10, dtype=torch.float64),) * 2
    code = torch.randint(10, (500, 2))
    x = TableTensor(
        numerical=numerical,
        categorical=CategoricalTensor(code=code, categories=categories),
    )
    y = TableTensor.from_tensor(torch.randn(500, 1))

    output, output_y = Coarsen(
        columns=1.0,
        num_levels=LogUniformInt(4, 4),
        grid_probability=0.0,
        categorical=True,
        standardize=True,
    )(x=x, y=y)

    assert (
        output.numerical[:, -1]
        .nan_to_num(9)
        .equal(numerical[:, -1].nan_to_num(9))
    )
    for column in output.numerical[:, :-1].mT:
        assert 2 <= column.unique().numel() <= 4
        torch.testing.assert_close(column.mean(), column.new_tensor(0.0))
        torch.testing.assert_close(column.std(), column.new_tensor(1.0))
    for index, category in enumerate(output.categorical.categories):
        assert 2 <= len(category) <= 4
        values = category[output.categorical.code[:, index]]
        torch.testing.assert_close(values.mean(), values.new_tensor(0.0))
        torch.testing.assert_close(values.std(), values.new_tensor(1.0))
    assert output_y.numerical.equal(y.numerical)


def test_coarsen_gives_one_category_for_zero() -> None:
    x = TableTensor(
        categorical=CategoricalTensor(
            code=torch.arange(3).repeat(10)[:, None],
            categories=(torch.tensor([-0.3, 0.3, 1.2]),),
        ),
    )
    y = TableTensor.from_tensor(torch.randn(30, 1))

    output, _ = Coarsen(
        columns=1.0,
        num_levels=LogUniformInt(3, 3),
        step=LogUniform(1.0, 1.0),
        grid_probability=1.0,
        categorical=True,
    )(x=x, y=y, generator=torch.Generator().manual_seed(0))

    assert output.categorical.categories[0].tolist() == [0.0, 1.0]
