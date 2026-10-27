# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import pytest
import torch

from sdm import EnsembleTable, TableTensor
from sdm.processing import SelectColumns


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
    ).fit_transform_ensemble(EnsembleTable.from_table(table, num_members=4))

    expected_columns = (
        ("num_0", "num_1"),
        ("num_2", "num_3"),
        ("num_4", "num_0"),
        ("num_1", "num_2"),
    )
    for member_id, columns in enumerate(expected_columns):
        indices = [table.numerical_columns.index(column) for column in columns]
        expected = TableTensor.from_tensor(
            table.numerical[..., indices],
            columns=columns,
        )
        assert out[member_id].equal(expected)


@pytest.mark.parametrize("method", ["first", "round_robin"])
def test_select_columns_preserves_mixed_datetime_units(
    method: Literal["first", "round_robin"],
) -> None:
    table = TableTensor(
        columns={"datetime": ("us", "ns", "other_us")},
        datetime=torch.tensor([[1, 1_001, 2], [2, 2_001, 3]]),
        datetime_units=("us", "ns", "us"),
    )
    output = SelectColumns(
        max_columns=2, method=method
    ).fit_transform_ensemble(EnsembleTable.from_table(table, num_members=3))
    columns = (
        [("us", "ns")] * 3
        if method == "first"
        else [("us", "ns"), ("other_us", "us"), ("ns", "other_us")]
    )
    original = table.to_arrow()
    for i, names in enumerate(columns):
        assert output[i].to_arrow().equals(original.select(names))
