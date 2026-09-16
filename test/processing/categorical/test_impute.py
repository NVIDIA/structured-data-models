# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.testing import withCUDA


def _table(
    values: list[list[int]] | list[list[list[int]]],
    *,
    categories: tuple[tuple[str, ...], ...] = (
        ("a", "b", "c"),
        ("x", "y"),
    ),
    device: torch.device | None = None,
) -> TableTensor:
    return TableTensor(
        categorical=CategoricalTensor(
            code=torch.tensor(values, dtype=torch.int32, device=device),
            categories=tuple(
                StringTensor.from_list(category, device=device)
                for category in categories
            ),
        ),
    )


@withCUDA
def test_impute_mode_uses_most_frequent_category(
    device: torch.device,
) -> None:
    context = _table(
        [
            [[0, 1], [0, -1], [1, 1], [-1, 0]],
            [[2, 0], [2, 0], [1, 1], [-1, 1]],
        ],
        device=device,
    )
    query = _table(
        [
            [[-1, -1], [2, 0]],
            [[-1, -1], [0, 1]],
        ],
        device=device,
    )
    processor = sp.ImputeMode().fit(context)

    output = processor.transform(query)

    assert torch.equal(
        processor._fill_values,
        torch.tensor([[[0, 1]], [[2, 0]]], device=device),
    )
    assert torch.equal(
        output.categorical.code,
        torch.tensor(
            [[[0, 1], [2, 0]], [[2, 0], [0, 1]]],
            dtype=torch.int32,
            device=device,
        ),
    )
    assert output.columns[Stype.categorical] == ("cat_0", "cat_1")
    for actual, expected in zip(
        output.categorical.categories,
        query.categorical.categories,
    ):
        assert torch.equal(actual, expected)


@withCUDA
def test_impute_mode_tie_uses_lowest_code(
    device: torch.device,
) -> None:
    table = _table([[1, 0], [0, 1], [-1, -1]], device=device)

    output = sp.ImputeMode().fit_transform(table)

    assert torch.equal(
        output.categorical.code,
        torch.tensor(
            [[1, 0], [0, 1], [0, 0]],
            dtype=torch.int32,
            device=device,
        ),
    )


def test_impute_mode_rejects_all_missing_column() -> None:
    table = _table([[0, -1], [1, -1]])

    with pytest.raises(ValueError, match=r"cat_1.*no observed values"):
        sp.ImputeMode().fit(table)


@pytest.mark.parametrize(
    "categories",
    [
        (("b", "a", "c"), ("x", "y")),
        (("a", "b"), ("x", "y")),
    ],
)
def test_impute_mode_rejects_changed_vocabulary(
    categories: tuple[tuple[str, ...], ...],
) -> None:
    processor = sp.ImputeMode().fit(_table([[0, 0], [0, 1]]))
    query = _table([[-1, -1]], categories=categories)

    with pytest.raises(
        ValueError,
        match=r"vocabulary.*cat_0.*fitted values.*AlignCategories",
    ):
        processor.transform(query)


def test_impute_mode_rejects_reordered_columns() -> None:
    processor = sp.ImputeMode().fit(_table([[0, 0], [0, 1]]))
    query = TableTensor(
        columns={"categorical": ("segment", "kind")},
        categorical=CategoricalTensor(
            code=torch.tensor([[-1, -1]], dtype=torch.int32),
            categories=(
                StringTensor.from_list(["x", "y"]),
                StringTensor.from_list(["a", "b", "c"]),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"vocabulary.*segment.*fitted values",
    ):
        processor.transform(query)


def test_impute_mode_rejects_out_of_range_code_during_fit() -> None:
    table = _table([[3, 0], [0, 1]])

    with pytest.raises(ValueError, match=r"cat_0.*outside.*vocabulary"):
        sp.ImputeMode().fit(table)


def test_impute_mode_rejects_out_of_range_code_during_transform() -> None:
    processor = sp.ImputeMode().fit(_table([[0, 0], [1, 1]]))
    query = _table([[3, -1]])

    with pytest.raises(ValueError, match=r"cat_0.*outside.*vocabulary"):
        processor.transform(query)


def test_impute_mode_composes_before_to_numerical() -> None:
    table = _table([[0, 0], [0, -1], [1, 1], [-1, 1]])
    processor = sp.StypeDispatch(
        categorical=[sp.ImputeMode(), sp.ToNumerical()],
    )

    output = processor.fit_transform(table)

    assert output.columns[Stype.numerical] == ("cat_0", "cat_1")
    assert output.columns[Stype.categorical] == ()
    assert torch.equal(
        output.numerical,
        torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0]],
        ),
    )
