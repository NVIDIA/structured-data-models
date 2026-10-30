# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch
from torch import Tensor

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    NaT,
    NullableTensor,
    StringTensor,
    Stype,
    StypeLike,
    TableTensor,
)
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("value", [0, -1, 2.5, True, float("nan")])
@pytest.mark.parametrize("tensor_value", [False, True])
@pytest.mark.parametrize("mask_shape", [(), (3, 1), (1, 2), (2, 3, 2)])
def test_numerical(
    device: torch.device,
    value: float,
    tensor_value: bool,
    mask_shape: tuple[int, ...],
) -> None:
    values = torch.arange(6, device=device, dtype=torch.float64).view(3, 2)
    table = TableTensor(numerical=values)
    mask = torch.ones(mask_shape, device=device, dtype=torch.bool)
    fill = torch.tensor(value, device=device) if tensor_value else value

    out = table.masked_fill(mask, fill)
    assert isinstance(out, TableTensor)
    expected = values.masked_fill(mask, fill)
    torch.testing.assert_close(out.numerical, expected, equal_nan=True)
    torch.testing.assert_close(
        values, torch.arange(6, device=device).view(3, 2).double()
    )
    assert out.numerical.dtype == values.dtype
    assert out.columns == table.columns
    assert out.categorical.size(-1) == 0


@withCUDA
@pytest.mark.parametrize("tensor_value", [False, True])
def test_category_values(device: torch.device, tensor_value: bool) -> None:
    codes = torch.tensor([[0, 1], [1, -1], [-1, 0]], device=device)
    categories = (
        torch.tensor([10, 20], device=device),
        torch.tensor([20, 30], device=device),
    )
    tensor = CategoricalTensor(code=codes, categories=categories)
    mask = torch.tensor([[True], [False], [True]], device=device)
    value = torch.tensor(20, device=device) if tensor_value else 20

    out = torch.masked_fill(tensor, mask, value)
    assert isinstance(out, CategoricalTensor)

    torch.testing.assert_close(
        out.code,
        torch.tensor([[1, 0], [1, -1], [1, 0]], device=device),
    )
    for actual, expected in zip(out.categories, categories):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(tensor.code, codes)


@pytest.mark.parametrize("value", [0, -1, 20.5])
def test_unknown_category(value: float) -> None:
    tensor = CategoricalTensor(
        code=torch.tensor([[0], [1]]), categories=(torch.tensor([10, 20]),)
    )
    with pytest.raises(RuntimeError, match="existing category"):
        tensor.masked_fill(torch.tensor([[True], [False]]), value)

    out = tensor.masked_fill(torch.zeros(2, 1, dtype=torch.bool), value)
    assert isinstance(out, CategoricalTensor)
    torch.testing.assert_close(out.code, tensor.code)


def test_nullable_category_values() -> None:
    tensor = CategoricalTensor(
        code=torch.tensor([[0], [-1]]),
        categories=(
            NullableTensor(
                data=torch.tensor([10, 20]), valid=torch.tensor([True, False])
            ),
        ),
    )
    mask = torch.tensor([[False], [True]])
    out = tensor.masked_fill(mask, 10)
    assert isinstance(out, CategoricalTensor)
    assert out.code[:, 0].tolist() == [0, 0]
    with pytest.raises(RuntimeError, match="existing category"):
        tensor.masked_fill(mask, 20)


@withCUDA
@pytest.mark.parametrize("tensor_value", [False, True])
def test_missing_string_categories(
    device: torch.device, tensor_value: bool
) -> None:
    tensor = CategoricalTensor(
        code=torch.tensor([[0], [1], [-1]], device=device),
        categories=(StringTensor.from_list(["a", "b"], device=device),),
    )
    fill = (
        torch.tensor(torch.nan, device=device) if tensor_value else torch.nan
    )
    out = tensor.masked_fill(
        torch.tensor([[True], [False], [False]], device=device), fill
    )
    assert isinstance(out, CategoricalTensor)
    assert out.code[:, 0].tolist() == [-1, 1, -1]
    assert out.categories[0].tolist() == ["a", "b"]


@withCUDA
@pytest.mark.parametrize("reverse_schema", [False, True])
def test_table_column_order(
    device: torch.device, reverse_schema: bool
) -> None:
    columns: dict[StypeLike, tuple[str, ...]] = {
        Stype.numerical: ("a", "b"),
        Stype.categorical: ("kind",),
    }
    if reverse_schema:
        columns = dict(reversed(tuple(columns.items())))
    table = TableTensor(
        columns=columns,
        numerical=torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], device=device),
            categories=(torch.tensor([10, 20], device=device),),
        ),
    )
    mask = torch.tensor(
        [[False, True, True], [True, False, False]], device=device
    )
    out = torch.masked_fill(table, mask, 20)
    assert isinstance(out, TableTensor)
    torch.testing.assert_close(
        out.numerical, torch.tensor([[1.0, 20.0], [20.0, 4.0]], device=device)
    )
    assert out.categorical.code[:, 0].tolist() == [1, 1]
    assert out.columns == table.columns


@withCUDA
@pytest.mark.parametrize("tensor_value", [False, True])
def test_missing_all_stypes(device: torch.device, tensor_value: bool) -> None:
    table = TableTensor(
        numerical=torch.tensor([[1.0], [2.0]], device=device),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], device=device),
            categories=(torch.tensor([10, 20], device=device),),
        ),
        datetime=torch.tensor([[100], [200]], device=device),
        text=StringTensor.from_list([["a"], ["b"]], device=device),
        id=ColumnarTensor(
            columns=[
                torch.tensor([1, 2], device=device),
                StringTensor.from_list(["x", "y"], device=device),
            ]
        ),
    )
    fill = (
        torch.tensor(torch.nan, device=device) if tensor_value else torch.nan
    )
    out = table.masked_fill(
        torch.tensor([[True], [False]], device=device), fill
    )
    assert isinstance(out, TableTensor)
    assert out.numerical[:, 0].isnan().tolist() == [True, False]
    assert out.categorical.code[:, 0].tolist() == [-1, 1]
    assert out.datetime[:, 0].tolist() == [NaT, 200]
    assert out.text[:, 0].tolist() == [None, "b"]
    assert out.id[:, 0].tolist() == [None, 2]
    assert out.id[:, 1].tolist() == [None, "y"]
    assert table.text[:, 0].tolist() == ["a", "b"]
    assert table.id[:, 0].tolist() == [1, 2]
    assert out.columns == table.columns


@withCUDA
@pytest.mark.parametrize("tensor_value", [False, True])
def test_nullable_fill(device: torch.device, tensor_value: bool) -> None:
    tensor = NullableTensor(
        data=torch.tensor([1, 2, 3], device=device),
        valid=torch.tensor([False, True, False], device=device),
    )
    value = torch.tensor(-1, device=device) if tensor_value else -1
    out = tensor.masked_fill(
        torch.tensor([True, False, False], device=device), value
    )
    assert isinstance(out, NullableTensor)
    assert out.tolist() == [-1, 2, None]
    assert tensor.tolist() == [None, 2, None]


@withCUDA
def test_noncontiguous_text(device: torch.device) -> None:
    tensor = (
        StringTensor.from_list(["a", "b", "c", None], device=device)
        .view(2, 2)
        .T
    )
    mask = torch.tensor([[False, True], [True, False]], device=device)
    out = tensor.masked_fill(mask, torch.nan)
    assert isinstance(out, StringTensor)
    assert out.tolist() == [["a", None], [None, None]]
    assert tensor.tolist() == [["a", "c"], ["b", None]]


@withCUDA
def test_gradients(device: torch.device) -> None:
    values = torch.arange(6.0, device=device).view(3, 2).requires_grad_()
    table = TableTensor(numerical=values)
    fill = torch.tensor(2.0, device=device, requires_grad=True)
    mask = torch.tensor([[True], [False], [True]], device=device)
    out = table.masked_fill(mask, fill)
    assert isinstance(out, TableTensor)
    out.numerical.sum().backward()
    torch.testing.assert_close(values.grad, (~mask).expand_as(values).float())
    torch.testing.assert_close(fill.grad, fill.new_tensor(4.0))


@withCUDA
def test_empty_table(device: torch.device) -> None:
    table = TableTensor(size=(3,), device=device)
    out = table.masked_fill(
        torch.ones(2, 3, 1, dtype=torch.bool, device=device), 0
    )
    assert isinstance(out, TableTensor)
    assert out.shape == (2, 3, 0)
    assert out.columns == table.columns


@pytest.mark.parametrize(
    "mask", [torch.ones(2, 1), torch.ones(2, 3, dtype=torch.bool)]
)
def test_invalid_masks(mask: Tensor) -> None:
    table = TableTensor(numerical=torch.ones(2, 1))
    with pytest.raises(RuntimeError, match=r"boolean mask|column dimension"):
        table.masked_fill(mask, 0)


def test_nonscalar_fill() -> None:
    table = TableTensor(numerical=torch.ones(2, 1))
    with pytest.raises(RuntimeError, match="scalar fill value"):
        table.masked_fill(torch.ones(2, 1, dtype=torch.bool), torch.ones(2))


def test_text_rejects_numeric_fill() -> None:
    tensor = StringTensor.from_list(["a", "b"])
    with pytest.raises(RuntimeError, match="variable-length cells"):
        tensor.masked_fill(torch.tensor([True, False]), 0)


@withCUDA
@pytest.mark.parametrize("tensor_value", [False, True])
def test_datetime_and_integer_fills(
    device: torch.device, tensor_value: bool
) -> None:
    table = TableTensor(
        numerical=torch.tensor([[1], [2]], dtype=torch.int32, device=device),
        datetime=torch.tensor([[100], [200]], device=device),
        id=ColumnarTensor(columns=[torch.tensor([10, 20], device=device)]),
    )
    fill = torch.tensor(-1, device=device) if tensor_value else -1
    out = table.masked_fill(
        torch.tensor([[True], [False]], device=device), fill
    )
    assert isinstance(out, TableTensor)
    assert out.numerical[:, 0].tolist() == [-1, 2]
    assert out.numerical.dtype == torch.int32
    assert out.datetime[:, 0].tolist() == [-1, 200]
    assert out.id[:, 0].tolist() == [-1, 20]


@withCUDA
def test_compiled(device: torch.device, compilation: None) -> None:
    table = TableTensor(
        numerical=torch.tensor([[1.0], [2.0]], device=device),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], device=device),
            categories=(torch.tensor([10, 20], device=device),),
        ),
    )
    mask = torch.tensor([[True], [False]], device=device)

    def fill(table: TableTensor, mask: Tensor, value: Tensor) -> TableTensor:
        return cast(TableTensor, table.masked_fill(mask, value))

    compiled = torch.compile(fill, backend="aot_eager", fullgraph=True)
    for value in (10.0, 20.0, torch.nan):
        scalar = torch.tensor(value, device=device)
        expected = fill(table, mask, scalar)
        actual = compiled(table, mask, scalar)
        torch.testing.assert_close(
            actual.numerical, expected.numerical, equal_nan=True
        )
        torch.testing.assert_close(
            actual.categorical.code, expected.categorical.code
        )
