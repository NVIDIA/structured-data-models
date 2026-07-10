import pytest
import torch
from sdm import TableTensor
from sdm.processing import Recipe, Sequential, StandardScale


def _table(numerical: torch.Tensor | None = None) -> TableTensor:
    if numerical is None:
        numerical = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    return TableTensor.from_tensor(numerical, columns=("x0", "x1"))


def test_recipe_normalizes_empty_roles() -> None:
    recipe = Recipe(features=[StandardScale()], target=None, output=[])

    assert isinstance(recipe.features, Sequential)
    assert isinstance(recipe.target, Sequential)
    assert isinstance(recipe.output, Sequential)
    assert len(recipe.features.steps) == 1
    assert len(recipe.target.steps) == 0
    assert len(recipe.output.steps) == 0


@pytest.mark.parametrize(
    "recipe",
    [
        Recipe(),
        Recipe(
            features=[StandardScale()],
            target=[StandardScale(), StandardScale()],
            output=[],
        ),
        Recipe(
            features=StandardScale(),
            target=StandardScale(),
            output=StandardScale(),
        ),
    ],
)
def test_recipe_repr_indents_roles(recipe: Recipe) -> None:
    description = repr(recipe)

    assert description.startswith("Recipe(\n  features=")
    assert "\n  target=" in description
    assert "\n  output=" in description
    assert description.endswith("\n)")


def test_target_forward_then_inverse_round_trips() -> None:
    recipe = Recipe(target=[StandardScale()])
    table = _table()

    assert isinstance(recipe.target, Sequential)
    transformed = recipe.target.fit_transform(table)
    restored = recipe.target.inverse_transform(transformed)

    assert not torch.equal(transformed.numerical, table.numerical)
    assert torch.allclose(restored.numerical, table.numerical, atol=1e-6)


def test_recipe_roles_fit_transform_features_and_target() -> None:
    recipe = Recipe(features=[StandardScale()], target=[StandardScale()])
    features = _table()
    target = _table(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))

    out_features = recipe.features.fit_transform(features)
    out_target = recipe.target.fit_transform(target)

    assert torch.allclose(
        out_features.numerical.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )
    assert torch.allclose(
        out_target.numerical.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )


def test_recipe_role_fit_accepts_table() -> None:
    recipe = Recipe(features=[StandardScale()])
    features = _table()

    fitted = recipe.features.fit(features)
    transformed = recipe.features.transform(features)

    assert fitted is recipe.features
    assert torch.allclose(
        transformed.numerical.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )
