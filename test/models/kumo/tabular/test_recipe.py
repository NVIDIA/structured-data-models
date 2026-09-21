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


def _transform(
    features: TableTensor,
    num_members: int = 2,
    numerical_missing: str = "nan",
) -> EnsembleTable:
    recipe = default_recipe(numerical_missing)  # type: ignore[arg-type]
    return recipe.features.fit_transform_ensemble(
        EnsembleTable.from_table(features, num_members=num_members)
    )


@withCUDA
def test_default_recipe_keeps_missing_values_of_small_tables(
    device: torch.device,
) -> None:
    features = _features(100, 3, 2, device=device)
    numerical = features.numerical.clone()
    numerical[::7, 1] = float("nan")
    numerical[3, 2] = float("inf")
    features = features.replace_blocks(numerical=numerical)

    output = _transform(features)

    for member_id in range(output.num_members):
        member = output.table(member_id)
        assert member.numerical.isnan().sum() == numerical.isnan().sum()
        assert not member.numerical.isinf().any()
        assert member.numerical.dtype == features.numerical.dtype


@withCUDA
def test_dispatch_recipe_imputes_small_tables(device: torch.device) -> None:
    features = _features(100, 3, 2, device=device)
    numerical = features.numerical.clone()
    numerical[::7, 1] = float("nan")
    numerical[3, 2] = float("inf")
    features = features.replace_blocks(numerical=numerical)

    output = _transform(features, numerical_missing="dispatch")

    for member_id in range(output.num_members):
        member = output.table(member_id)
        assert member.numerical.isfinite().all()
        assert member.numerical.dtype == features.numerical.dtype


@withCUDA
def test_dispatch_recipe_keeps_missing_values_of_large_sparse_tables(
    device: torch.device,
) -> None:
    features = _features(20_000, 3, 2, device=device)
    numerical = features.numerical.clone()
    numerical[::5, 0] = float("nan")
    numerical[1::5, 1] = float("nan")
    numerical[2::5, 2] = float("nan")
    features = features.replace_blocks(numerical=numerical)

    output = _transform(features, numerical_missing="dispatch")

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


def test_default_recipe_regression_reduction_options() -> None:
    def regression_steps(recipe: sp.Recipe) -> list[type]:
        dispatcher = next(
            m
            for m in recipe.output.modules()
            if isinstance(m, sp.TaskDispatch)
        )
        return [
            type(m)
            for m in dispatcher.processors["regression"].modules()
            if isinstance(
                m, (sp.SortQuantiles, sp.ReduceQuantiles, sp.ReduceEstimators)
            )
        ]

    assert regression_steps(default_recipe()) == [
        sp.ReduceQuantiles,
        sp.ReduceEstimators,
    ]
    assert regression_steps(
        default_recipe(regression_reduction="quantile_trim")
    ) == [
        sp.SortQuantiles,
        sp.ReduceEstimators,
        sp.ReduceQuantiles,
    ]

    # Eight members, five rows, nine quantiles: the trimmed mean over the
    # members of every quantile, then the mean over the quantiles.
    recipe = default_recipe(regression_reduction="quantile_trim")
    for dispatcher in recipe.output.modules():
        if isinstance(dispatcher, sp.TaskDispatch):
            dispatcher._task = "regression"
    x = torch.randn(8, 5, 9)
    out = recipe.output.transform(TableTensor.from_tensor(x))
    assert out.size() == (5, 1)
    expected = (
        x.sort(dim=-1)
        .values.sort(dim=0)
        .values[1:-1]
        .mean(dim=0)
        .mean(dim=-1, keepdim=True)
    )
    torch.testing.assert_close(out.numerical, expected)


def test_default_recipe_numerical_missing_options() -> None:
    def kinds(recipe: sp.Recipe) -> set[type]:
        return {type(m) for m in recipe.features.modules()}

    assert sp.MissingDispatch in kinds(default_recipe("dispatch"))
    assert sp.MissingDispatch not in kinds(default_recipe())
    assert sp.MissingDispatch not in kinds(default_recipe("nan"))
    assert sp.ImputeMean not in kinds(default_recipe("nan"))
    mix = [
        m
        for m in default_recipe("mix").features.modules()
        if isinstance(m, sp.Choice)
    ]
    assert any(
        {type(getattr(o, "processor", o)) for o in choice.options}
        == {sp.Identity, sp.ImputeMean}
        for choice in mix
    )
    assert sp.MissingDispatch not in kinds(default_recipe("impute"))
    assert sp.ImputeMean in kinds(default_recipe("impute"))
