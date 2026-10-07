# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    Stype,
    TableTensor,
)


@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_compile_table(fullgraph: bool, mixed: bool) -> None:
    torch.compiler.reset()

    def transform(table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical.square())[1::2]

    compiled = torch.compile(transform, fullgraph=fullgraph, dynamic=True)
    for rows in (5, 9, 0, 5):
        numerical = torch.arange(rows * 8.0).reshape(rows, 8)[:, 1::2]
        blocks: dict[str, Any] = {}
        if mixed:
            blocks = {
                "categorical": CategoricalTensor(
                    code=torch.arange(rows).remainder(2).view(rows, 1),
                    categories=(torch.tensor([3, 7]),),
                ),
                "datetime": torch.arange(rows).view(rows, 1),
                "text": StringTensor.from_list(["value"] * rows).view(rows, 1),
                "id": ColumnarTensor((torch.arange(rows),)),
            }
        table = TableTensor(numerical=numerical, **blocks)
        inputs = (table,) if mixed else (table, table[..., 1::2, :])
        for inp in inputs:
            actual, expected = compiled(inp), transform(inp)
            assert actual.columns == expected.columns
            assert torch.equal(actual, expected)


@pytest.mark.parametrize("fullgraph", [False, True])
def test_compile_table_construction(fullgraph: bool) -> None:
    torch.compiler.reset()

    def transform(x: torch.Tensor) -> TableTensor:
        return TableTensor.from_tensor(x.sin())

    compiled = torch.compile(transform, fullgraph=fullgraph, dynamic=True)
    for rows in (5, 9, 0):
        x = torch.randn(rows, 6)[:, 1::2]
        torch.testing.assert_close(compiled(x).numerical, x.sin())


@pytest.mark.parametrize("fullgraph", [False, True])
def test_compile_table_view_mutation(fullgraph: bool) -> None:
    torch.compiler.reset()

    def transform(table: TableTensor) -> TableTensor:
        view = table[1::2]
        view.numerical.add_(3)
        return view

    compiled = torch.compile(transform, fullgraph=fullgraph, dynamic=True)
    for rows in (5, 9, 0):
        x = torch.arange(rows * 8.0).reshape(rows, 8)[:, 1::2]
        table = TableTensor(numerical=x)
        expected = TableTensor(numerical=x.clone())
        result = compiled(table)
        assert torch.equal(result, transform(expected))
        assert torch.equal(table, expected)
        result.numerical.add_(5)
        assert torch.equal(result.numerical, table.numerical[1::2])


@pytest.mark.parametrize("fullgraph", [False, True])
def test_compile_table_schema_guards(fullgraph: bool) -> None:
    torch.compiler.reset()

    def transform(table: TableTensor) -> torch.Tensor:
        columns = dict(table._column_items)
        scale = 2 if columns[Stype.numerical][0] == "a" else 3
        return table.numerical * scale

    compiled = torch.compile(transform, fullgraph=fullgraph, dynamic=True)
    for names, rows in (
        (["a", "b"], 5),
        (["c", "d"], 5),
        (["b", "a"], 5),
        (["a", "b"], 9),
    ):
        table = TableTensor(
            columns={"numerical": names},
            numerical=torch.randn(rows, 2),
        )
        assert all(isinstance(key, Stype) for key in table.columns)
        torch.testing.assert_close(compiled(table), transform(table))
