# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
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


def test_default_recipe_keeps_values_distinct_beside_outliers() -> None:
    numerical = torch.arange(100.0).unsqueeze(-1)
    numerical[0] = 1e12

    output = KumoTabular.default_recipe().features.fit_transform_ensemble(
        EnsembleTable.from_table(
            TableTensor.from_tensor(numerical),
            num_members=8,
        )
    )

    members = [output[member_id] for member_id in range(len(output))]
    assert all(member.numerical.dtype == torch.float32 for member in members)
    num_unique = [member.numerical[1:].unique().numel() for member in members]
    assert max(num_unique) == 99


@pytest.mark.parametrize("cardinality", [50, 51])
def test_default_recipe_adds_category_counts(cardinality: int) -> None:
    codes = torch.cat(
        [torch.arange(cardinality), torch.zeros(300 - cardinality)]
    )
    features = TableTensor(
        numerical=torch.randn(300, 2),
        categorical=CategoricalTensor(
            code=codes.long().unsqueeze(-1),
            categories=(torch.arange(cardinality),),
        ),
    )

    output = KumoTabular.default_recipe().features.fit_transform_ensemble(
        EnsembleTable.from_table(features, num_members=2)
    )

    expected_columns = {"num_0", "num_1", "cat_0"}
    if cardinality > 50:
        expected_columns.add("cat_0__count")
    for member_id in range(len(output)):
        assert (
            set(output[member_id].columns[Stype.numerical]) == expected_columns
        )


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


@withCUDA
@pytest.mark.parametrize("num_members", [1, 4, 8, 16])
def test_default_recipe_numerical_schedule(
    num_members: int, device: torch.device
) -> None:
    values = torch.arange(1, 101, device=device).double().square()
    values[7] = torch.nan
    features = TableTensor.from_tensor(values.unsqueeze(-1))
    recipe = KumoTabular.default_recipe()
    output = recipe.features.fit_transform_ensemble(
        EnsembleTable.from_table(features, num_members=num_members)
    )
    query = features[:13]
    prediction_features = recipe.features.transform_ensemble(
        EnsembleTable.from_table(query, num_members=num_members)
    )
    for i in range(num_members):
        processor = sp.Sequential(
            sp.Standardize(eps=1e-6),
            sp.Clip(-100.0, 100.0),
            (
                sp.Identity(),
                sp.PowerTransform(),
                [sp.RobustScale(), sp.ClipSoft(3.0)],
                sp.RankGaussian(),
            )[i % 4],
            sp.ClipSigma(threshold=4.0),
        )
        expected = processor.fit_transform(features).numerical.float()
        # Independent sign flips are allowed; magnitudes identify each view.
        torch.testing.assert_close(
            output[i].numerical.abs(), expected.abs(), equal_nan=True
        )
        torch.testing.assert_close(
            prediction_features[i].numerical,
            output[i].numerical[:13],
            equal_nan=True,
        )


@withCUDA
@pytest.mark.parametrize("num_members", [1, 8, 16])
@pytest.mark.parametrize("num_classes", [2, 3, 10])
def test_default_recipe_balances_class_shifts(
    num_members: int, num_classes: int, device: torch.device
) -> None:
    target = TableTensor(
        categorical=CategoricalTensor(
            code=torch.arange(num_classes, device=device).unsqueeze(-1),
            categories=(torch.arange(num_classes, device=device),),
        )
    )
    output = KumoTabular.default_recipe().target.fit_transform_ensemble(
        EnsembleTable.from_table(target, num_members=num_members)
    )
    codes = torch.stack(
        [output[i].categorical.code[0, 0] for i in range(num_members)]
    )
    counts = codes.long().bincount(minlength=num_classes)
    assert counts.max() - counts.min() <= 1
    for i in range(num_members):
        assert output[i].categorical.tolist() == target.categorical.tolist()
