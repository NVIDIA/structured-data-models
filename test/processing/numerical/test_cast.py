# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.processing import Cast


def test_cast() -> None:
    table = TableTensor(
        numerical=torch.randn(5, 2),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [0], [1], [1]]),
            categories=(torch.arange(2),),
        ),
    )

    out = Cast(torch.float64).transform(table)

    assert out.numerical.dtype == torch.float64
    torch.testing.assert_close(out.numerical, table.numerical.double())
    assert out.categorical.code.equal(table.categorical.code)
    assert out.columns == table.columns
    assert Cast(torch.float32).transform(out).numerical.dtype == torch.float32
    assert repr(Cast(torch.float64)) == "Cast(torch.float64)"


def test_cast_leaves_tables_without_numerical_columns() -> None:
    table = TableTensor(
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]]),
            categories=(torch.arange(2),),
        ),
    )
    out = Cast(torch.float64).transform(table)
    assert out.columns[Stype.numerical] == ()
    assert out.categorical.code.equal(table.categorical.code)
