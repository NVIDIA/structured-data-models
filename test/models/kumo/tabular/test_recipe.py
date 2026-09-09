import torch

from sdm import CategoricalTensor, TableTensor
from sdm.models.kumo.tabular import KumoTabular
from sdm.tensor import EnsembleTable
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
        EnsembleTable(features, num_members=2)
    )

    for member_id in range(output.num_members):
        member = output.table(member_id)
        assert member.numerical.isnan().sum() == 3
        assert not member.numerical.isinf().any()
