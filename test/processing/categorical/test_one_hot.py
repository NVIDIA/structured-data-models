# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, TableTensor
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_one_hot_values_and_columns(
    device: torch.device, batch_shape: tuple[int, ...]
) -> None:
    codes = torch.tensor(
        [[0, -1, 1], [1, -1, 0], [-1, -1, -1]],
        device=device,
        dtype=torch.int32,
    ).expand(*batch_shape, 3, 3)
    table = TableTensor(
        columns={"categorical": ("color", "empty", "size")},
        categorical=CategoricalTensor(
            code=codes,
            categories=(
                torch.tensor([10, 20], device=device),
                torch.empty(0, device=device, dtype=torch.long),
                torch.tensor([50, 60, 70], device=device),
            ),
        ),
    )

    out = sp.OneHotEncode().transform(table)

    expected = torch.tensor(
        [[1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]],
        device=device,
        dtype=torch.float32,
    ).expand(*batch_shape, 3, 5)
    torch.testing.assert_close(out.numerical, expected)
    assert out.numerical_columns == (
        "color__one_hot_0",
        "color__one_hot_1",
        "size__one_hot_0",
        "size__one_hot_1",
        "size__one_hot_2",
    )
    assert out.categorical_columns == ()


def test_one_hot_alignment_uses_context_vocabulary() -> None:
    context = TableTensor.from_pandas(
        pd.DataFrame({"color": ["red", "blue", "red"]}),
        stypes={"color": "categorical"},
    )
    query = TableTensor.from_pandas(
        pd.DataFrame({"color": ["blue", "red", "green", None]}),
        stypes={"color": "categorical"},
    )
    processor = sp.Sequential(sp.AlignCategories(), sp.OneHotEncode())
    fitted = processor.fit_transform(context)
    actual = processor.transform(query)

    torch.testing.assert_close(
        actual.numerical,
        torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    )
    assert actual.numerical_columns == fitted.numerical_columns


@pytest.mark.parametrize(("rows", "categories"), [(0, 0), (3, 0), (0, 2)])
def test_one_hot_empty_inputs(rows: int, categories: int) -> None:
    table = TableTensor(
        categorical=CategoricalTensor(
            code=torch.full((rows, 1), -1),
            categories=(torch.arange(categories),),
        ),
    )

    out = sp.OneHotEncode().transform(table)

    assert out.numerical.shape == (rows, categories)
    assert out.categorical.shape == (rows, 0)
