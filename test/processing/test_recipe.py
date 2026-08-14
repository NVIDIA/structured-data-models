import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.models import TabICLv2
from sdm.testing import withCUDA


def _table(numerical: torch.Tensor | None = None) -> TableTensor:
    if numerical is None:
        numerical = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    return TableTensor.from_tensor(numerical)


def test_recipe_normalizes_empty_roles_and_repr() -> None:
    recipe = sp.Recipe(features=[sp.Standardize()], target=None, output=[])

    assert isinstance(recipe.features, sp.Sequential)
    assert isinstance(recipe.output, sp.Sequential)
    assert len(recipe.features) == 1
    assert len(recipe.output) == 0
    assert "features=Sequential" in repr(recipe)
    assert "target=Identity()" in repr(recipe)


def test_target_forward_then_inverse_round_trips() -> None:
    recipe = sp.Recipe(target=[sp.Standardize()])
    table = _table()

    assert isinstance(recipe.target, sp.Sequential)
    transformed = recipe.target.fit_transform(table)
    restored = recipe.target.inverse_transform(transformed)

    assert not torch.equal(transformed.numerical, table.numerical)
    assert torch.allclose(restored.numerical, table.numerical, atol=1e-6)


def test_recipe_roles_fit_transform_features_and_target() -> None:
    recipe = sp.Recipe(
        features=[sp.Standardize()],
        target=[sp.Standardize()],
    )
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


def test_recipe_rejects_stateful_processor_shared_across_roles() -> None:
    processor = sp.Standardize()

    with pytest.raises(
        ValueError,
        match=r"Standardize.*Recipe\.features.*Recipe\.target",
    ):
        sp.Recipe(features=processor, target=processor)


def test_recipe_rejects_nested_stateful_alias() -> None:
    processor = sp.Standardize()

    with pytest.raises(
        ValueError,
        match=r"Standardize.*Recipe\.features.*Recipe\.target",
    ):
        sp.Recipe(
            features=[sp.Identity(), processor],
            target=[processor, sp.Identity()],
        )


def test_recipe_rejects_stateful_processor_reused_within_role() -> None:
    processor = sp.Standardize()

    with pytest.raises(
        ValueError,
        match=(
            r"Standardize.*Recipe\.features\.0\.processor"
            r".*Recipe\.features\.1\.processor"
        ),
    ):
        sp.Recipe(features=[processor, processor])


def test_recipe_allows_stateless_processor_shared_across_roles() -> None:
    processor = sp.Identity()
    recipe = sp.Recipe(features=processor, target=processor)
    table = _table()

    torch.testing.assert_close(
        recipe.features.transform(table).numerical,
        table.numerical,
    )
    torch.testing.assert_close(
        recipe.target.transform(table).numerical,
        table.numerical,
    )


def test_recipe_alias_validation_does_not_replace_existing_role() -> None:
    processor = sp.Standardize()
    recipe = sp.Recipe(features=processor)
    target = recipe.target

    with pytest.raises(ValueError, match=r"reused at"):
        recipe.append_target(processor)

    assert recipe.target is target


def test_recipe_role_fit_accepts_table() -> None:
    recipe = sp.Recipe(features=[sp.Standardize()])
    features = _table()

    fitted = recipe.features.fit(features)
    transformed = recipe.features.transform(features)

    assert fitted is recipe.features
    assert torch.allclose(
        transformed.numerical.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )


def test_recipe_rejects_task_dispatch_in_target() -> None:
    with pytest.raises(ValueError, match=r"not supported.*Recipe.target"):
        sp.Recipe(target=sp.TaskDispatch(regression=sp.Identity()))


@withCUDA
def test_tabiclv2_default_recipe_on_device(device: torch.device) -> None:
    recipe = TabICLv2.default_recipe()

    features = TableTensor(
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
    target = TableTensor.from_tensor(torch.randn(8, 1, device=device))

    model_features = recipe.features.fit_transform(features)
    model_target = recipe.target.fit_transform(target)

    assert model_features.size() == features.size()
    assert model_features.numerical.device == device
    assert model_features.categorical.size(-1) == 0
    assert set(model_features.columns[Stype.numerical]) == {
        "num_0",
        "num_1",
        "cat_0",
    }
    assert torch.isfinite(model_features.numerical).all()

    assert model_target.numerical.device == device
    assert isinstance(recipe.target, sp.InvertibleMixin)
    restored = recipe.target.inverse_transform(model_target)
    torch.testing.assert_close(restored.numerical, target.numerical)

    # The original categorical column uses sparse codes 1 and 3.
    transformed = TabICLv2.default_recipe().target.fit_transform(
        features.select_columns("cat_0")
    )
    assert transformed.categorical.unique().sort().values.tolist() == [0, 1]
