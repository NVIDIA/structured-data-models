# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import cast

import pytest
import torch

import sdm.processing as sp
from sdm import EnsembleTable, Recipe, TableTensor
from sdm.processing.execution import RecipeExecution
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("smooth", [0.0, 1.0, 5.0])
def test_target_encoding_regression(
    device: torch.device, smooth: float
) -> None:
    x = TableTensor.from_columns(
        data={"city": ["a", "a", "b", None], "number": [1.0, 2.0, 3.0, 4.0]},
        stypes={"city": "categorical", "number": "numerical"},
        device=device,
    )
    y = TableTensor.from_tensor(
        x.numerical.new_tensor([[0.0], [2.0], [6.0], [8.0]])
    )
    processor = sp.AddTargetEncoding(smooth=smooth)
    execution = RecipeExecution(Recipe(features=processor))
    (context,) = execution.fit_transform(x=x, y=y, related_tables=None)

    # Four folds leave one row out, regardless of the random permutation.
    expected = x.numerical.new_tensor(
        [
            (2 + smooth * (16 / 3)) / (1 + smooth),
            (0 + smooth * (14 / 3)) / (1 + smooth),
            10 / 3,
            8 / 3,
        ]
    )
    torch.testing.assert_close(context.x.numerical[:, -1], expected)
    assert context.x.categorical.equal(x.categorical)
    torch.testing.assert_close(context.x.numerical[:, :1], x.numerical)

    # Independent tensorization changes vocabulary order and adds a new value.
    query = TableTensor.from_columns(
        data={
            "city": ["b", "a", "unseen", None],
            "number": [5.0, 6.0, 7.0, 8.0],
        },
        stypes={"city": "categorical", "number": "numerical"},
        device=device,
    )
    (out,) = execution.transform(x=query, related_tables=None)
    expected = x.numerical.new_tensor(
        [
            (6 + smooth * 4) / (1 + smooth),
            (2 + smooth * 4) / (2 + smooth),
            4.0,
            (8 + smooth * 4) / (1 + smooth),
        ]
    )
    torch.testing.assert_close(out.x.numerical[:, -1], expected)
    assert out.x.numerical_columns == ("number", "city__target")


@withCUDA
@pytest.mark.parametrize("labels", [[9, 2, 9, 2, 2, 2], [9, 2, 5, 9, 2, 5]])
@pytest.mark.parametrize("smooth", [0.0, 2.0])
def test_target_encoding_classification(
    device: torch.device, labels: list[int], smooth: float
) -> None:
    x = TableTensor.from_columns(
        data={"city": ["a", "a", "a", "b", "b", "b"]},
        stypes={"city": "categorical"},
        device=device,
    )
    y = TableTensor.from_columns(
        data={"label": labels}, stypes={"label": "categorical"}, device=device
    )
    execution = RecipeExecution(
        Recipe(
            features=sp.AddTargetEncoding(smooth=smooth),
            target=sp.ShuffleCategories(method="shift"),
        )
    )
    contexts = execution.fit_transform(
        x=x, y=y, related_tables=None, num_members=3
    )
    assert all(context.x.equal(contexts[0].x) for context in contexts)
    outputs = execution.transform(x=x, related_tables=None)
    values = torch.tensor(labels, device=device)
    indicators = (values[:, None] == values.unique(sorted=True)).float()
    if indicators.size(-1) == 2:
        indicators = indicators[:, 1:]
    expected = torch.cat(
        [
            (
                (block.sum(0) + smooth * indicators.mean(0)) / (3 + smooth)
            ).expand(3, -1)
            for block in indicators.split(3)
        ]
    )
    for out in outputs:
        torch.testing.assert_close(out.x.numerical, expected)
    if expected.size(-1) == 3:
        torch.testing.assert_close(expected.sum(-1), expected.new_ones(6))


@withCUDA
def test_target_encoding_excludes_entire_fold(device: torch.device) -> None:
    x = TableTensor.from_columns(
        data={"city": ["a"] * 12},
        stypes={"city": "categorical"},
        device=device,
    )
    values = torch.arange(12, device=device, dtype=torch.float32)[:, None]
    # A fixed split is necessary to compare the same held-out rows.
    seed = 31
    held = torch.randperm(
        12, device=device, generator=torch.Generator(device).manual_seed(seed)
    )[:4]
    results = []
    for changed in [False, True]:
        target = values.clone()
        if changed:
            target[held] += 1e20
        execution = RecipeExecution(
            Recipe(features=sp.AddTargetEncoding(smooth=2, num_folds=3))
        )
        (context,) = execution.fit_transform(
            x=x,
            y=TableTensor.from_tensor(target),
            related_tables=None,
            generator=torch.Generator(device).manual_seed(seed),
        )
        results.append(context.x.numerical)
    torch.testing.assert_close(results[0][held], results[1][held])
    assert not torch.equal(results[0], results[1])


@withCUDA
def test_target_encoding_batch_and_state(device: torch.device) -> None:
    x = cast(
        TableTensor,
        TableTensor.from_columns(
            data={"city": ["a", "a", "b", None]},
            stypes={"city": "categorical"},
            device=device,
        ).expand(2, 3, 4, 1),
    )
    values = torch.arange(24, device=device, dtype=torch.float32).reshape(
        2, 3, 4, 1
    )
    y = TableTensor.from_tensor(values)
    processor = sp.AddTargetEncoding(smooth=1)
    execution = RecipeExecution(Recipe(features=processor))
    (context,) = execution.fit_transform(
        x=x, y=y, related_tables=None, num_members=1
    )
    assert context.x.size() == (2, 3, 4, 2)
    (expected,) = execution.transform(x=x, related_tables=None)

    state = deepcopy(processor.state_dict())
    restored = sp.AddTargetEncoding(smooth=1).to(device)
    restored.load_state_dict(state)
    torch.testing.assert_close(
        restored.transform(x).numerical, expected.x.numerical
    )
    restored.to(dtype=torch.float64)
    actual = restored.transform(cast(TableTensor, x.to(dtype=torch.float64)))
    torch.testing.assert_close(
        actual.numerical, expected.x.numerical.double(), check_dtype=True
    )
    for i in range(2):
        for j in range(3):
            mean = values[i, j].mean()
            expected_values = torch.stack(
                [
                    (values[i, j, :2].sum() + mean) / 3,
                    (values[i, j, :2].sum() + mean) / 3,
                    (values[i, j, 2, 0] + mean) / 2,
                    (values[i, j, 3, 0] + mean) / 2,
                ]
            )
            torch.testing.assert_close(
                expected.x.numerical[i, j, :, 0], expected_values
            )


@withCUDA
def test_target_encoding_distinct_member_targets(device: torch.device) -> None:
    x = TableTensor.from_columns(
        data={"city": ["a", "a", "b", "b"]},
        stypes={"city": "categorical"},
        device=device,
    )
    y = TableTensor.from_tensor(torch.arange(4.0, device=device)[:, None])
    processor = sp.AddTargetEncoding()
    processor._target = EnsembleTable.from_tables(
        tables=[y, TableTensor.from_tensor(y.numerical + 10)],
        member_table_ids=[0, 1, 0],
    )
    processor.fit_ensemble(EnsembleTable.from_table(x, num_members=3))
    query = EnsembleTable.from_tables(
        tables=[x, x[:2]], member_table_ids=[0, 1, 0]
    )
    outputs = processor.transform_ensemble(query)
    torch.testing.assert_close(
        outputs[0].numerical,
        y.numerical.new_tensor([[0.5], [0.5], [2.5], [2.5]]),
    )
    torch.testing.assert_close(
        outputs[1].numerical, y.numerical.new_full((2, 1), 10.5)
    )
    assert outputs[0].equal(outputs[2])


@withCUDA
def test_target_encoding_multiple_columns_and_empty_route(
    device: torch.device,
) -> None:
    x = TableTensor.from_columns(
        data={"city": ["a", "a", "b", "b"], "missing": [None] * 4},
        stypes={"city": "categorical", "missing": "categorical"},
        device=device,
    )
    numbers = TableTensor.from_tensor(
        torch.arange(4.0, device=device)[:, None]
    )
    features = EnsembleTable.from_tables(
        tables=[x, numbers], member_table_ids=[0, 1, 0]
    )
    processor = sp.AddTargetEncoding()
    processor._target = EnsembleTable.from_table(numbers, num_members=3)
    context = processor.fit_transform_ensemble(features)
    expected = numbers.numerical.new_tensor(
        [
            [1.0, 2.0],
            [0.0, 5 / 3],
            [3.0, 4 / 3],
            [2.0, 1.0],
        ]
    )
    torch.testing.assert_close(context[0].numerical, expected)
    assert context[1].equal(numbers)
    assert context[0].equal(context[2])
    out = processor.transform_ensemble(features)
    expected = numbers.numerical.new_tensor(
        [
            [0.5, 1.5],
            [0.5, 1.5],
            [2.5, 1.5],
            [2.5, 1.5],
        ]
    )
    torch.testing.assert_close(out[0].numerical, expected)
    assert out[1].equal(numbers)


@withCUDA
def test_target_encoding_class_absent_from_training_fold(
    device: torch.device,
) -> None:
    x = TableTensor.from_columns(
        data={"city": ["a"] * 3},
        stypes={"city": "categorical"},
        device=device,
    )
    y = TableTensor.from_columns(
        data={"label": ["z", "b", "m"]},
        stypes={"label": "categorical"},
        device=device,
    )
    execution = RecipeExecution(
        Recipe(features=sp.AddTargetEncoding(smooth=1))
    )
    (context,) = execution.fit_transform(x=x, y=y, related_tables=None)
    expected = context.x.numerical.new_tensor(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]
    )
    torch.testing.assert_close(context.x.numerical, expected)
