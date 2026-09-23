# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.processing import AddCategoryCounts
from sdm.testing import withCUDA


def _table(
    codes: list[list[int]] | list[list[list[int]]],
    device: torch.device | str = "cpu",
) -> TableTensor:
    return TableTensor(
        columns={Stype.categorical: ("city", "kind")},
        categorical=CategoricalTensor(
            code=torch.tensor(codes, dtype=torch.int32, device=device),
            categories=(
                torch.arange(4, device=device),
                torch.arange(2, device=device),
            ),
        ),
    )


@withCUDA
def test_add_category_counts_appends_log1p_of_fitted_counts(
    device: torch.device,
) -> None:
    # 'city' observes counts 3, 2, 1 and one missing value, 'kind' 4 and 3.
    context = _table(
        [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [-1, 1]],
        device=device,
    )

    output = AddCategoryCounts().fit_transform(context)

    assert output.columns[Stype.categorical] == ("city", "kind")
    assert output.columns[Stype.numerical] == ("city__count", "kind__count")
    assert output.categorical.equal(context.categorical)
    torch.testing.assert_close(
        output.numerical,
        torch.tensor(
            [
                [3.0, 4.0],
                [3.0, 4.0],
                [3.0, 3.0],
                [2.0, 4.0],
                [2.0, 3.0],
                [1.0, 4.0],
                [1.0, 3.0],
            ],
            device=device,
        ).log1p(),
    )


@withCUDA
def test_add_category_counts_uses_fitted_counts(device: torch.device) -> None:
    context = _table(
        [[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [-1, 1]],
        device=device,
    )
    # Code 3 is part of the vocabulary but was never observed while fitting.
    query = _table([[2, 0], [0, 1], [-1, 0], [3, 1]], device=device)

    output = AddCategoryCounts().fit(context).transform(query)

    torch.testing.assert_close(
        output.numerical,
        torch.tensor(
            [[1.0, 4.0], [3.0, 3.0], [1.0, 4.0], [0.0, 3.0]],
            device=device,
        ).log1p(),
    )


@withCUDA
@pytest.mark.parametrize("min_cardinality", [0, 3])
def test_add_category_counts_fits_leading_batches_independently(
    device: torch.device,
    min_cardinality: int,
) -> None:
    batched = _table(
        [
            [[0, 0], [0, 1], [1, 0], [2, 1]],
            [[0, 0], [1, 1], [1, 0], [2, 1]],
        ],
        device=device,
    )

    output = AddCategoryCounts(min_cardinality=min_cardinality).fit_transform(
        batched,
    )

    assert output.size() == (2, 4, 4 if min_cardinality == 0 else 3)
    torch.testing.assert_close(
        output.numerical[..., 0],
        torch.tensor(
            [[2.0, 2.0, 1.0, 1.0], [1.0, 2.0, 2.0, 1.0]],
            device=device,
        ).log1p(),
    )


def test_add_category_counts_avoids_float16_count_overflow() -> None:
    num_rows = 70_000
    context = TableTensor(
        columns={Stype.categorical: ("city",)},
        numerical=torch.empty(num_rows, 0, dtype=torch.float16),
        categorical=CategoricalTensor(
            code=torch.zeros(num_rows, 1, dtype=torch.int32),
            categories=(torch.tensor([0]),),
        ),
    )

    output = AddCategoryCounts().fit_transform(context)

    expected = torch.tensor(num_rows, dtype=torch.float32).log1p().half()
    torch.testing.assert_close(
        output.numerical[:, 0],
        expected.expand(num_rows),
    )


@withCUDA
@pytest.mark.parametrize(
    ("min_cardinality", "columns"),
    [
        (0, ("city__count", "kind__count")),
        (2, ("city__count",)),
        (3, ("city__count",)),
        (4, ()),
        (5, ()),
    ],
)
def test_add_category_counts_selects_by_fitted_vocabulary_size(
    device: torch.device,
    min_cardinality: int,
    columns: tuple[str, ...],
) -> None:
    # City has four vocabulary entries but only three observed categories.
    context = _table([[0, 0], [0, 1], [1, 0], [2, 1], [-1, -1]], device)
    query = _table([[0, 1], [0, 1]], device)
    processor = AddCategoryCounts(min_cardinality=min_cardinality).fit(context)

    output = processor.transform(query)

    assert output.columns[Stype.numerical] == columns
    assert output.categorical.equal(query.categorical)
    expected = torch.full((2, len(columns)), 2.0, device=device).log1p()
    torch.testing.assert_close(output.numerical, expected)

    restored = AddCategoryCounts(min_cardinality=min_cardinality)
    restored.load_state_dict(processor.state_dict())
    assert restored.transform(query).equal(output)
