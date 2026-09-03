# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import Stype, TableTensor
from sdm.processing import SelectColumns, SelectRows
from sdm.tensor import EnsembleTable


def test_select_columns_rejects_negative_max_columns() -> None:
    with pytest.raises(ValueError, match="max_columns must be positive"):
        SelectColumns(max_columns=-1)


def test_select_first_columns() -> None:
    table = TableTensor(
        numerical=torch.arange(6, dtype=torch.float).view(2, 3),
        datetime=torch.arange(6, dtype=torch.int64).view(2, 3),
    )

    out = SelectColumns(max_columns=2, method="first").transform(table)
    assert out.equal(table.select_columns(("num_0", "num_1", "dt_0", "dt_1")))


def test_select_columns_round_robin_routes_members() -> None:
    table = TableTensor(
        numerical=torch.arange(10, dtype=torch.float).view(2, 5),
    )

    out = SelectColumns(
        max_columns=2,
        method="round_robin",
    ).fit_transform_ensemble(EnsembleTable(table, num_members=4))

    expected_columns = (
        ("num_0", "num_1"),
        ("num_2", "num_3"),
        ("num_4", "num_0"),
        ("num_1", "num_2"),
    )
    for member_id, columns in enumerate(expected_columns):
        indices = [
            table.columns[Stype.numerical].index(column) for column in columns
        ]
        expected = TableTensor.from_tensor(
            table.numerical[..., indices],
            columns=columns,
        )
        assert out.table(member_id).equal(expected)


@pytest.mark.parametrize("max_rows", [0, -1])
def test_select_rows_rejects_non_positive_max_rows(max_rows: int) -> None:
    with pytest.raises(ValueError, match="max_rows must be positive"):
        SelectRows(max_rows=max_rows)


def test_select_first_rows() -> None:
    table = TableTensor.from_tensor(torch.randn(2, 3, 2))
    out = SelectRows(max_rows=2, method="first").transform(table)
    assert out.equal(table[:, :2])


def test_select_rows_round_robin_routes_members() -> None:
    table = TableTensor.from_tensor(torch.randn(5, 2))

    out = SelectRows(
        max_rows=2,
        method="round_robin",
    ).transform_ensemble(EnsembleTable(table, num_members=4))

    expected_rows = ((0, 1), (2, 3), (4, 0), (1, 2))
    for member_id, rows in enumerate(expected_rows):
        assert out.table(member_id).equal(table[list(rows)])


@pytest.mark.parametrize("num_rows", [0, 3])
def test_select_rows_round_robin_keeps_all_rows_within_limit(
    num_rows: int,
) -> None:
    table = TableTensor.from_tensor(torch.randn(num_rows, 2))

    out = SelectRows(
        max_rows=3,
        method="round_robin",
    ).transform_ensemble(EnsembleTable(table, num_members=2))

    assert all(out.table(member_id).equal(table) for member_id in range(2))
