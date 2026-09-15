import torch

import sdm.processing as sp
from sdm import CategoricalTensor, EnsembleTable, Stype, TableTensor
from sdm.models.kumo.tabular import KumoTabular
from sdm.models.kumo.tabular.recipe import default_recipe
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

    for member_id in range(output.num_members):
        member = output.table(member_id)
        expected_missing = torch.tensor(
            [
                missing_by_column[column]
                for column in member.columns[Stype.numerical]
            ],
            device=device,
        ).T
        assert torch.equal(member.numerical.isnan(), expected_missing)
        assert not member.numerical.isinf().any()


def test_default_recipe_numerical_missing() -> None:
    def count(recipe: sp.Recipe, cls: type) -> int:
        return sum(isinstance(p, cls) for p in recipe.features.modules())

    assert count(default_recipe(), sp.ImputeMean) == 0
    assert count(default_recipe("impute"), sp.ImputeMean) == 1
    assert count(default_recipe("impute"), sp.Choice) == 1
    assert count(default_recipe("mix"), sp.ImputeMean) == 1
    assert count(default_recipe("mix"), sp.Choice) == 2


def test_default_recipe_variants() -> None:
    def count(recipe: sp.Recipe, cls: type) -> int:
        return sum(isinstance(p, cls) for p in recipe.features.modules())

    assert count(default_recipe(numeric_transform="power"), sp.Choice) == 0
    assert (
        count(
            default_recipe(numeric_transform="quantile"), sp.QuantileTransform
        )
        == 1
    )
    assert (
        count(default_recipe(shuffle_categories_max=30), sp.ShuffleCategories)
        == 1
    )
    assert (
        count(default_recipe(interactions=True), sp.PairwiseInteractions) == 1
    )
