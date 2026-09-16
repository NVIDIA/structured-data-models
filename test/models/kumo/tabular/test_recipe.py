# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import sdm.processing as sp
from sdm import CategoricalTensor, EnsembleTable, Stype, TableTensor
from sdm.models.kumo.tabular import KumoTabular
from sdm.models.kumo.tabular.recipe import default_recipe
from sdm.testing import withCUDA


def _features(
    num_rows: int,
    num_numerical: int,
    num_levels: int,
    device: torch.device | str = "cpu",
) -> TableTensor:
    return TableTensor(
        numerical=torch.randn(num_rows, num_numerical, device=device),
        categorical=CategoricalTensor(
            code=torch.randint(num_levels, (num_rows, 1), device=device),
            categories=(torch.arange(num_levels, device=device),),
        ),
    )


def _transform(features: TableTensor, num_members: int = 2) -> EnsembleTable:
    recipe = KumoTabular.default_recipe()
    return recipe.features.fit_transform_ensemble(
        EnsembleTable.from_table(features, num_members=num_members)
    )


@withCUDA
def test_default_recipe_imputes_small_tables(device: torch.device) -> None:
    features = _features(100, 3, 2, device=device)
    numerical = features.numerical.clone()
    numerical[::7, 1] = float("nan")
    numerical[3, 2] = float("inf")
    features = features.replace_blocks(numerical=numerical)

    output = _transform(features)

    for member_id in range(output.num_members):
        member = output.table(member_id)
        assert member.numerical.isfinite().all()
        assert member.numerical.dtype == features.numerical.dtype


@withCUDA
def test_default_recipe_keeps_missing_values_of_large_sparse_tables(
    device: torch.device,
) -> None:
    features = _features(20_000, 3, 2, device=device)
    numerical = features.numerical.clone()
    numerical[::5, 0] = float("nan")
    numerical[1::5, 1] = float("nan")
    numerical[2::5, 2] = float("nan")
    features = features.replace_blocks(numerical=numerical)

    output = _transform(features)

    for member_id in range(output.num_members):
        member = output.table(member_id)
        assert member.numerical.isnan().sum() == numerical.isnan().sum()
        assert not member.numerical.isinf().any()


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    return torch.corrcoef(torch.stack((x, y)))[0, 1].item()


def test_default_recipe_flips_numbers_but_not_codes() -> None:
    features = _features(300, 32, 3)
    output = _transform(features, num_members=8)

    code_correlations = []
    numerical_correlations = []
    for member_id in range(output.num_members):
        member = output.table(member_id)
        for index, column in enumerate(member.columns[Stype.numerical]):
            values = member.numerical[:, index]
            if column == "cat_0":
                reference = features.categorical.code[:, 0].float()
                code_correlations.append(_correlation(values, reference))
            else:
                reference = features[column].numerical[:, 0]
                numerical_correlations.append(_correlation(values, reference))

    # Codes pass through monotone normalizations only; numbers are also
    # sign-flipped per member, so some columns end up anti-correlated.
    assert len(code_correlations) == output.num_members
    assert min(code_correlations) > 0
    assert min(numerical_correlations) < 0 < max(numerical_correlations)


def test_default_recipe_appends_level_counts() -> None:
    features = _features(300, 2, 60)
    output = _transform(features)

    member = output.table(0)
    assert set(member.columns[Stype.numerical]) == {
        "num_0",
        "num_1",
        "cat_0",
        "cat_0__count",
    }
    assert member.numerical.isfinite().all()


def test_default_recipe_reduces_outputs_per_task() -> None:
    recipe = KumoTabular.default_recipe()
    dispatchers = [
        module
        for module in recipe.output.modules()
        if isinstance(module, sp.TaskDispatch)
    ]

    for dispatcher in dispatchers:
        dispatcher._task = "regression"
    out = recipe.output.transform(
        TableTensor.from_tensor(torch.randn(8, 5, 9))
    )
    assert out.size() == (5, 1)
    assert out.columns[Stype.numerical] == ("mean",)

    for dispatcher in dispatchers:
        dispatcher._task = "classification"
    out = recipe.output.transform(
        TableTensor.from_tensor(torch.randn(8, 5, 3))
    )
    assert out.size() == (5, 3)
    torch.testing.assert_close(out.numerical.sum(dim=-1), torch.ones(5))


def test_default_recipe_variants() -> None:
    def count(recipe: sp.Recipe, cls: type) -> int:
        return sum(isinstance(p, cls) for p in recipe.features.modules())

    assert count(default_recipe(), sp.Choice) == 2
    assert count(default_recipe(normalize="power"), sp.Choice) == 0
    assert (
        count(default_recipe(normalize="quantile"), sp.QuantileTransform) == 2
    )
    assert (
        count(default_recipe(shuffle_categories_max=30), sp.ShuffleCategories)
        == 1
    )
