# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch

from sdm import CategoricalTensor, EnsembleTable, TableTensor
from sdm.processing import DropHighCardinalityColumns
from sdm.testing import withCUDA


@withCUDA
def test_drop_high_cardinality_columns(device: torch.device) -> None:
    table = TableTensor.from_columns(
        data={
            "low": ["a", "b", "a", None, "b", "a"],
            "high": ["a", "b", "c", "a", None, "c"],
            "missing": [None] * 6,
            "number": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "text": ["a", "b", "c", "d", "e", "f"],
        },
        stypes={
            "low": "categorical",
            "high": "categorical",
            "missing": "categorical",
            "number": "numerical",
            "text": "text",
        },
        device=device,
    )
    processor = DropHighCardinalityColumns(threshold=3)
    output = processor.fit_transform(table)
    assert output.equal(table.drop_columns("high"))

    # Keep the fitted schema even when the query has different cardinalities.
    query = TableTensor.from_columns(
        data={
            "low": ["u", "v", "w", "x"],
            "high": ["a"] * 4,
            "missing": ["u", "v", "w", "x"],
            "number": [0.0, 1.0, 2.0, 3.0],
            "text": ["a", "b", "c", "d"],
        },
        stypes={
            "low": "categorical",
            "high": "categorical",
            "missing": "categorical",
            "number": "numerical",
            "text": "text",
        },
        device=device,
    )
    assert processor.transform(query).equal(query.drop_columns("high"))


@withCUDA
def test_drop_high_cardinality_counts_observed_values(
    device: torch.device,
) -> None:
    table = TableTensor.from_columns(
        data={"category": ["a", "a", "b", "c", "d", "e"]},
        stypes={"category": "categorical"},
        device=device,
    )
    context = table[:3]
    processor = DropHighCardinalityColumns(threshold=3)
    assert processor.fit_transform(context).equal(context)
    assert processor.transform(table).equal(table)

    # Refitting learns a new selection from the new training rows.
    assert processor.fit_transform(table).categorical_columns == ()


@withCUDA
def test_drop_high_cardinality_ensemble(device: torch.device) -> None:
    first = TableTensor.from_columns(
        data={"a": ["x", "y", "z", "x"], "b": ["x"] * 4},
        stypes={"a": "categorical", "b": "categorical"},
        device=device,
    )
    # Use shared vocabularies with enough entries for both columns.
    categories = (first.categorical.categories[0],) * 2
    second = first.replace_blocks(
        categorical=CategoricalTensor(
            code=first.categorical.code.flip(-1),
            categories=categories,
        )
    )
    first = first.replace_blocks(
        categorical=CategoricalTensor(
            code=first.categorical.code,
            categories=categories,
        )
    )
    context = EnsembleTable(
        groups=[cast(TableTensor, torch.stack([first, second]))],
        locations=[(0, 1), (0, 0), (0, 1)],
    )
    processor = DropHighCardinalityColumns(threshold=3)
    output = processor.fit_transform_ensemble(context)
    query = processor.transform_ensemble(
        EnsembleTable.from_table(first, num_members=3)
    )
    for member_id in range(3):
        reference = DropHighCardinalityColumns(threshold=3).fit(
            context[member_id]
        )
        assert output[member_id].equal(reference.transform(context[member_id]))
        assert query[member_id].equal(reference.transform(first))
    assert output[0].categorical_columns == ("a",)
    assert output[1].categorical_columns == ("b",)
