# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
import torch
from sklearn.preprocessing import OneHotEncoder

from sdm import CategoricalTensor, ColumnarTensor, TableTensor
from sdm.processing import AlignCategories, OneHotEncode, Sequential
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16]
)
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_one_hot_values_and_columns(
    device: torch.device, dtype: torch.dtype, batch_shape: tuple[int, ...]
) -> None:
    codes = (
        torch.tensor(
            [[0, 1], [1, 0], [-1, -1]], device=device, dtype=torch.int32
        )
        .expand(*batch_shape, 3, 2)
        .clone()
    )
    if batch_shape:
        codes[1] = codes[1].flip(-2)
    numerical = torch.tensor(
        [[2.0], [4.0], [6.0]], device=device, dtype=dtype
    ).expand(*batch_shape, 3, 1)
    table = TableTensor(
        columns={"numerical": ("value",), "categorical": ("color", "size")},
        numerical=numerical,
        categorical=CategoricalTensor(
            code=codes,
            categories=(
                torch.tensor([10, 20], device=device),
                torch.tensor([50, 60, 70], device=device),
            ),
        ),
        id=ColumnarTensor(
            (torch.arange(3, device=device).expand(*batch_shape, 3),)
        ),
    )

    out = OneHotEncode().transform(table)

    expected = (
        numerical.new_tensor(
            [[2, 1, 0, 0, 1, 0], [4, 0, 1, 1, 0, 0], [6, 0, 0, 0, 0, 0]]
        )
        .expand(*batch_shape, 3, 6)
        .clone()
    )
    if batch_shape:
        expected[1, ..., 1:] = expected[1, ..., 1:].flip(-2)
    torch.testing.assert_close(out.numerical, expected)
    assert out.device == device
    assert out.numerical_columns == (
        "value",
        "color__one_hot_0",
        "color__one_hot_1",
        "size__one_hot_0",
        "size__one_hot_1",
        "size__one_hot_2",
    )
    assert out.categorical_columns == ()
    assert out.id.equal(table.id)
    torch.testing.assert_close(table.numerical, numerical)
    torch.testing.assert_close(table.categorical.code, codes)


def test_one_hot_alignment_uses_context_vocabulary() -> None:
    context = TableTensor.from_pandas(
        pd.DataFrame({"color": ["red", "blue", "red"]}),
        stypes={"color": "categorical"},
    )
    query = TableTensor.from_pandas(
        pd.DataFrame({"color": ["blue", "red", "green", None]}),
        stypes={"color": "categorical"},
    )
    processor = Sequential(AlignCategories(), OneHotEncode())
    fitted = processor.fit_transform(context)
    actual = processor.transform(query)

    torch.testing.assert_close(
        fitted.numerical, torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    )
    torch.testing.assert_close(
        actual.numerical,
        torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    )
    assert actual.numerical_columns == fitted.numerical_columns

    restored = Sequential(AlignCategories(), OneHotEncode())
    restored.load_state_dict(processor.state_dict())
    assert restored.transform(query).equal(actual)


def test_one_hot_matches_sklearn() -> None:
    context = pd.DataFrame(
        {"color": ["red", "blue", "red"], "size": [2, 1, 3]}
    )
    query = pd.DataFrame({"color": ["blue", "green"], "size": [3, 9]})
    stypes = {"color": "categorical", "size": "categorical"}
    processor = Sequential(AlignCategories(sort_by="value"), OneHotEncode())
    processor.fit(TableTensor.from_pandas(context, stypes=stypes))
    actual = processor.transform(TableTensor.from_pandas(query, stypes=stypes))
    reference = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    reference.fit(context)
    expected = torch.from_numpy(reference.transform(query))
    torch.testing.assert_close(actual.numerical, expected, check_dtype=False)


@pytest.mark.parametrize("rows", [0, 3])
def test_one_hot_empty_vocabularies(rows: int) -> None:
    table = TableTensor(
        categorical=CategoricalTensor(
            code=torch.full((rows, 1), -1),
            categories=(torch.empty(0, dtype=torch.long),),
        ),
    )
    out = OneHotEncode().transform(table)
    assert out.numerical.shape == (rows, 0)
    assert out.categorical.shape == (rows, 0)


def test_one_hot_empty_query() -> None:
    table = TableTensor(
        categorical=CategoricalTensor(
            code=torch.empty((0, 1), dtype=torch.long),
            categories=(torch.tensor([10, 20]),),
        ),
    )
    out = OneHotEncode().transform(table)
    assert out.numerical.shape == (0, 2)


def test_one_hot_numerical_only_is_identity() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    assert OneHotEncode().transform(table) is table


def test_one_hot_mixed_empty_and_populated_vocabularies() -> None:
    table = TableTensor(
        categorical=CategoricalTensor(
            code=torch.tensor([[-1, 0], [-1, 1]]),
            categories=(
                torch.empty(0, dtype=torch.long),
                torch.tensor([3, 8]),
            ),
        ),
    )
    out = OneHotEncode().transform(table)
    assert out.numerical_columns == ("cat_1__one_hot_0", "cat_1__one_hot_1")
    torch.testing.assert_close(out.numerical, torch.eye(2))
