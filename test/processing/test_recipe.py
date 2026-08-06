import pytest
import torch

from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.models import TabICLv2
from sdm.processing import InvertibleMixin, Recipe, Sequential, Standardize
from sdm.testing import withCUDA


def _table(numerical: torch.Tensor | None = None) -> TableTensor:
    if numerical is None:
        numerical = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    return TableTensor.from_tensor(numerical, columns=("x0", "x1"))


def test_recipe_normalizes_empty_roles_and_repr() -> None:
    recipe = Recipe(features=[Standardize()], target=None, output=[])

    assert isinstance(recipe.features, Sequential)
    assert isinstance(recipe.target, Sequential)
    assert isinstance(recipe.output, Sequential)
    assert len(recipe.features) == 1
    assert len(recipe.target) == 0
    assert len(recipe.output) == 0
    assert "features=Sequential" in repr(recipe)
    assert "target=Sequential()" in repr(recipe)


def test_target_forward_then_inverse_round_trips() -> None:
    recipe = Recipe(target=[Standardize()])
    table = _table()

    assert isinstance(recipe.target, Sequential)
    transformed = recipe.target.fit_transform(table)
    restored = recipe.target.inverse_transform(transformed)

    assert not torch.equal(transformed.numerical, table.numerical)
    assert torch.allclose(restored.numerical, table.numerical, atol=1e-6)


def test_recipe_roles_fit_transform_features_and_target() -> None:
    recipe = Recipe(features=[Standardize()], target=[Standardize()])
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
    recipe = Recipe(features=[Standardize()])
    features = _table()

    fitted = recipe.features.fit(features)
    transformed = recipe.features.transform(features)

    assert fitted is recipe.features
    assert torch.allclose(
        transformed.numerical.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "fullgraph",
    [False, True],
    ids=["default", "fullgraph"],
)
@withCUDA
def test_tabiclv2_feature_recipe_compile(
    device: torch.device,
    fullgraph: bool,
) -> None:
    recipe = TabICLv2.default_recipe()
    features = TableTensor(
        columns={
            "numerical": ("a", "b"),
            "categorical": ("kind",),
        },
        numerical=torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 5.0],
                [7.0, 11.0],
                [13.0, 17.0],
            ],
            device=device,
        ),
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[0], [1], [0], [1]],
                dtype=torch.int32,
                device=device,
            ),
            categories=(StringTensor.from_list(["a", "b"], device=device),),
        ),
    )
    recipe.features.fit(features)
    expected = recipe.features.transform(features)

    transform = torch.compile(
        recipe.features.transform,
        backend="inductor",
        fullgraph=fullgraph,
    )
    actual = transform(features)

    torch.testing.assert_close(actual.numerical, expected.numerical)
    assert actual.columns == expected.columns


@withCUDA
def test_tabiclv2_default_recipe_on_device(device: torch.device) -> None:
    recipe = TabICLv2.default_recipe()

    features = TableTensor(
        columns={
            "numerical": ("a", "b"),
            "categorical": ("kind",),
        },
        numerical=torch.randn(8, 2, device=device),
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[1], [3]], dtype=torch.int32, device=device
            ).repeat(4, 1),
            categories=(
                StringTensor.from_list(
                    ["unused-a", "a", "unused-b", "b"],
                    device=device,
                ),
            ),
        ),
    )
    target = TableTensor.from_tensor(
        torch.randn(8, 1, device=device),
        columns=("y",),
    )

    model_features = recipe.features.fit_transform(features)
    model_target = recipe.target.fit_transform(target)

    assert model_features.size() == features.size()
    assert model_features.numerical.device == device
    assert model_features.categorical.size(-1) == 0
    assert set(model_features.columns[Stype.numerical]) == {"a", "b", "kind"}
    assert torch.isfinite(model_features.numerical).all()

    assert model_target.numerical.device == device
    assert isinstance(recipe.target, InvertibleMixin)
    restored = recipe.target.inverse_transform(model_target)
    torch.testing.assert_close(restored.numerical, target.numerical)

    # The original categorical column uses sparse codes 1 and 3.
    transformed = TabICLv2.default_recipe().target.fit_transform(
        features.select_columns("kind")
    )
    assert transformed.categorical.unique().sort().values.tolist() == [0, 1]
