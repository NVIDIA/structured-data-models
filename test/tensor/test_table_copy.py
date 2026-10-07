# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    NullableTensor,
    StringTensor,
    TableTensor,
)


def make_table() -> TableTensor:
    return TableTensor(
        numerical=torch.arange(12.0).reshape(3, 4)[:, 1::2],
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [0]]),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
        datetime=torch.arange(3).view(3, 1),
        text=StringTensor.from_list([["a"], ["bc"], ["def"]]),
        id=ColumnarTensor(
            (
                NullableTensor(
                    torch.arange(3), torch.tensor([True, False, True])
                ),
            )
        ),
    )


@pytest.mark.parametrize("fullgraph", [False, True])
def test_compile_mutate_mixed_table(fullgraph: bool) -> None:
    torch.compiler.reset()

    def mutate(table: TableTensor) -> TableTensor:
        table.numerical.add_(3)
        table.categorical.code.copy_(1 - table.categorical.code)
        table.datetime.add_(10)
        table.text._data.add_(1)
        cast(NullableTensor, table.id._columns[0])._data.add_(5)
        return table

    table, expected = make_table(), make_table()
    original = table.numerical
    result = torch.compile(mutate, fullgraph=fullgraph, dynamic=False)(table)
    mutate(expected)
    assert result is table
    assert torch.equal(table, expected)
    assert torch.equal(original, expected.numerical)


def test_copy_table_rejects_different_storage() -> None:
    table, source = make_table(), make_table()
    source.numerical.add_(100)
    source = source.replace_blocks(
        text=StringTensor.from_list([["longer"], ["bc"], ["def"]])
    )
    before = table.numerical.clone()
    with pytest.raises(ValueError, match="matching tensor storage shapes"):
        table.copy_(source)
    assert torch.equal(table.numerical, before)


def test_copy_table_preserves_shared_storage() -> None:
    table, source = make_table(), make_table()
    view = table[1:]
    source.numerical.add_(100)
    source.text._data.add_(1)
    assert table.copy_(source) is table
    assert torch.equal(table, source)
    assert torch.equal(view, source[1:])


@pytest.mark.parametrize("views", [False, True])
def test_copy_columnar_rejects_overlapping_storage(views: bool) -> None:
    data = torch.arange(8).reshape(2, 4)
    columns = tuple(data) if views else tuple(row.clone() for row in data)
    destination = ColumnarTensor(columns)
    source = ColumnarTensor(tuple(reversed(columns)))
    before = tuple(column.clone() for column in columns)
    with pytest.raises(
        ValueError,
        match=r"storage overlap|destination leaves sharing storage",
    ):
        destination.copy_(source)
    assert all(
        torch.equal(column, old) for column, old in zip(columns, before)
    )


@pytest.mark.parametrize("unchanged_second", [False, True])
def test_copy_columnar_rejects_aliased_destinations(
    unchanged_second: bool,
) -> None:
    data = torch.arange(4)
    alias = data.view_as(data)
    destination = ColumnarTensor((data, alias))
    source = ColumnarTensor(
        (data + 10, alias if unchanged_second else data + 20)
    )
    before = data.clone()
    with pytest.raises(ValueError, match="destination leaves sharing storage"):
        destination.copy_(source)
    assert torch.equal(data, before)
