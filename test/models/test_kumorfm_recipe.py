import torch
from sdm import ColumnarTensor, Stype, TableTensor
from sdm.models import KumoRFM
from sdm.models.tabiclv2.recipe import default_recipe


def test_default_recipe_matches_tabiclv2() -> None:
    recipe = KumoRFM.default_recipe()
    other = KumoRFM.default_recipe()

    assert repr(recipe) == repr(default_recipe())
    assert recipe is not other
    assert recipe.features is not other.features


def test_default_recipe_preserves_relational_ids() -> None:
    table = TableTensor(
        columns={
            Stype.numerical: ("value",),
            Stype.id: ("entity_id",),
        },
        numerical=torch.tensor([[1.0], [2.0]]),
        id=ColumnarTensor((torch.tensor([10, 11]),)),
    )

    transformed = KumoRFM.default_recipe().features.fit_transform(table)

    assert transformed.columns[Stype.id] == ("entity_id",)
    assert transformed.id.tolist() == table.id.tolist()
