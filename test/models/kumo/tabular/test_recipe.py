import torch

from sdm import CategoricalTensor, EnsembleTable, StringTensor, Stype, TableTensor
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
                [6.0, 6.0, 13.0],
            ],
            device=device,
        ),
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[0], [-1], [1], [0], [1], [2]], device=device
            ),
            categories=(torch.arange(3, device=device),),
        ),
    )
    recipe = KumoTabular.default_recipe()

    output = recipe.features.fit_transform_ensemble(
        EnsembleTable.from_table(features, num_members=2)
    )
    missing_by_column = {
        "num_0": [False, False, False, False, False, False],
        "num_1": [False, True, False, False, False, False],
        "num_2": [True, False, False, False, False, False],
        "cat_0": [False, False, False, False, False, True],
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
        categorical_index = member.columns[Stype.numerical].index("cat_0")
        assert member.numerical[1, categorical_index] == -1.0


def test_default_recipe_distinguishes_missing_from_unseen_categories() -> None:
    context = TableTensor(
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [0], [1], [-1]]),
            categories=(StringTensor.from_list(["seen_a", "seen_b"]),),
        ),
    )
    query = TableTensor(
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [-1]]),
            categories=(StringTensor.from_list(["unseen", "seen_a"]),),
        ),
    )
    recipe = KumoTabular.default_recipe()
    recipe.features.fit(context)

    output = recipe.features.transform(query)
    values = output.numerical[:, output.columns[Stype.numerical].index("cat_0")]

    assert values[0].isnan()
    assert values[1].isfinite()
    assert values[2] == -1.0
