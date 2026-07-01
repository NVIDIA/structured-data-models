import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import Recipe, StandardScale


def _table(numerical: torch.Tensor | None = None) -> TableTensor:
    if numerical is None:
        numerical = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    n_rows = numerical.shape[0]
    categorical = CategoricalTensor(
        data=(torch.arange(n_rows) % 2).unsqueeze(1),
        categories=(StringTensor.from_list(["a", "b"]),),
    )
    return TableTensor(
        columns={
            "numerical": ("x0", "x1"),
            "categorical": ("kind",),
        },
        numerical=numerical,
        categorical=categorical,
    )


def test_recipe_is_public_reexport() -> None:
    from sdm.processing.recipe import Recipe as CanonicalRecipe

    assert Recipe is CanonicalRecipe


def test_recipe_normalizes_empty_roles_and_repr() -> None:
    recipe = Recipe(features=[StandardScale()], target=None, output=[])

    assert len(recipe.features) == 1
    assert len(recipe.target) == 0
    assert len(recipe.output) == 0
    assert "features: StandardScale" in repr(recipe)
    assert "target: identity" in repr(recipe)


def test_target_forward_then_inverse_round_trips() -> None:
    recipe = Recipe(target=[StandardScale()])
    table = _table()

    transformed = recipe.target.fit_transform(table)
    restored = recipe.target.inverse_transform(transformed)

    assert not torch.equal(transformed.numerical, table.numerical)
    assert torch.allclose(restored.numerical, table.numerical, atol=1e-6)


def test_recipe_fit_transform_returns_features_and_target() -> None:
    recipe = Recipe(features=[StandardScale()], target=[StandardScale()])
    features = _table()
    target = _table(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))

    out_features, out_target = recipe.fit_transform(features, target)

    assert isinstance(out_features, TableTensor)
    assert isinstance(out_target, TableTensor)
    assert torch.allclose(
        out_features.numerical.mean(dim=0), torch.zeros(2), atol=1e-6
    )
    assert torch.allclose(
        out_target.numerical.mean(dim=0), torch.zeros(2), atol=1e-6
    )
