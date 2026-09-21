# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import sdm.processing as sp
from sdm import CategoricalTensor, EnsembleTable, Stype, TableTensor
from sdm.models.kumo.tabular import KumoTabular
from sdm.testing import withCUDA


@withCUDA
def test_default_recipe_preserves_missing_values(device: torch.device) -> None:
    features = TableTensor(
        numerical=torch.tensor(
            [
                [1.0, 1.0, float("inf")],
                [2.0, float("nan"), 5.0],
                [3.0, 3.0, 7.0],
                [4.0, 4.0, 9.0],
                [5.0, 5.0, 11.0],
            ],
            device=device,
        ),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [-1], [1], [0], [1]], device=device),
            categories=(torch.arange(2, device=device),),
        ),
    )
    recipe = KumoTabular.default_recipe()

    output = recipe.features.fit_transform_ensemble(
        EnsembleTable.from_table(features, num_members=2)
    )
    missing_by_column = {
        "num_0": [False, False, False, False, False],
        "num_1": [False, True, False, False, False],
        "num_2": [False, False, False, False, False],
        "cat_0": [False, False, False, False, False],
        "cat_0__count": [False, False, False, False, False],
    }

    for member_id in range(len(output)):
        member = output[member_id]
        expected_missing = torch.tensor(
            [
                missing_by_column[column]
                for column in member.columns[Stype.numerical]
            ],
            device=device,
        ).T
        assert torch.equal(member.numerical.isnan(), expected_missing)
        assert not member.numerical.isinf().any()


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    return torch.corrcoef(torch.stack((x, y)))[0, 1].item()


def test_default_recipe_flips_numbers_but_not_codes() -> None:
    features = TableTensor(
        numerical=torch.randn(300, 32),
        categorical=CategoricalTensor(
            code=torch.randint(3, (300, 1)),
            categories=(torch.arange(3),),
        ),
    )
    output = KumoTabular.default_recipe().features.fit_transform_ensemble(
        EnsembleTable.from_table(features, num_members=8)
    )

    code_correlations = []
    numerical_correlations = []
    for member_id in range(len(output)):
        member = output[member_id]
        for index, column in enumerate(member.columns[Stype.numerical]):
            values = member.numerical[:, index]
            if column == "cat_0":
                reference = features.categorical.code[:, 0].float()
                code_correlations.append(_correlation(values, reference))
            elif not column.endswith("__count"):
                reference = features[column].numerical[:, 0]
                numerical_correlations.append(_correlation(values, reference))

    assert len(code_correlations) == len(output)
    assert min(code_correlations) > 0
    assert min(numerical_correlations) < 0 < max(numerical_correlations)


def test_default_recipe_adds_category_counts() -> None:
    features = TableTensor(
        numerical=torch.randn(300, 2),
        categorical=CategoricalTensor(
            code=torch.randint(3, (300, 1)),
            categories=(torch.arange(3),),
        ),
    )

    output = KumoTabular.default_recipe().features.fit_transform_ensemble(
        EnsembleTable.from_table(features, num_members=2)
    )

    assert set(output[0].columns[Stype.numerical]) == {
        "num_0",
        "num_1",
        "cat_0",
        "cat_0__count",
    }


def test_default_recipe_reduces_outputs_per_task() -> None:
    recipe = KumoTabular.default_recipe()
    dispatch = next(
        module
        for module in recipe.output.modules()
        if isinstance(module, sp.TaskDispatch)
    )

    dispatch._task = "regression"
    output = recipe.output.transform(
        TableTensor.from_tensor(torch.randn(8, 5, 9))
    )
    assert output.size() == (5, 9)

    dispatch._task = "classification"
    output = recipe.output.transform(
        TableTensor.from_tensor(torch.randn(8, 5, 3))
    )
    assert output.size() == (5, 3)
    torch.testing.assert_close(output.numerical.sum(dim=-1), torch.ones(5))
