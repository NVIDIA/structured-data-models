# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, EnsembleTable, Stype, TableTensor
from sdm.processing import Cast, EnsembleProcessor
from sdm.testing import withCUDA


@withCUDA
def test_cast(device: torch.device) -> None:
    table = TableTensor(
        numerical=torch.randn(5, 2, device=device),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [0], [1], [1]], device=device),
            categories=(torch.arange(2, device=device),),
        ),
    )

    out = Cast(torch.float64).transform(table)

    assert not Cast(torch.float64).requires_fit
    assert out.numerical.dtype == torch.float64
    assert out.device == device
    torch.testing.assert_close(out.numerical, table.numerical.double())
    assert out.categorical is table.categorical
    assert out.columns == table.columns
    assert Cast(torch.float32).transform(out).numerical.dtype == torch.float32
    assert repr(Cast(torch.float64)) == "Cast(torch.float64)"


def test_cast_same_dtype_reuses_numerical_block() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    output = Cast(torch.float32).transform(table)
    assert output.numerical is table.numerical


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


def test_cast_through_ensemble_adapter() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0]], dtype=torch.float64))
    second = TableTensor.from_tensor(
        torch.tensor([[2.0]], dtype=torch.float64)
    )
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1),
    )
    output = EnsembleProcessor.as_processor(
        Cast(torch.float32)
    ).transform_ensemble(table)
    assert output.num_members == 2
    assert all(
        output.table(member_id).numerical.dtype == torch.float32
        for member_id in range(2)
    )
