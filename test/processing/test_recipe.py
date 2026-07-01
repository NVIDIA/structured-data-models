import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import (
    MeanImpute,
    Pipeline,
    Power,
    Recipe,
    SoftmaxTemperature,
    StandardScale,
)


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


def test_empty_pipeline_returns_input_table() -> None:
    table = _table()

    assert Pipeline().transform(table) is table
    assert Pipeline().fit_transform(table) is table
    assert Pipeline().inverse_transform(table) is table


def test_pipeline_transforms_numerical_and_passes_categorical() -> None:
    table = _table()

    output = Pipeline([StandardScale()]).fit_transform(table)

    assert not torch.equal(output.numerical, table.numerical)
    assert output.categorical is table.categorical
    assert output.columns == table.columns


def test_pipeline_rejects_non_processor_step() -> None:
    with pytest.raises(TypeError, match="Expected a Processor step"):
        Pipeline([object()])  # ty: ignore[invalid-argument-type]


def test_recipe_normalizes_empty_roles_and_repr() -> None:
    recipe = Recipe(features=[StandardScale()], target=None, output=[])

    assert len(recipe.features) == 1
    assert len(recipe.target) == 0
    assert len(recipe.output) == 0
    assert "features: StandardScale" in repr(recipe)
    assert "target: identity" in repr(recipe)


def test_pipeline_error_includes_step_position() -> None:
    # The second step is unfitted, so its transform raises with its position.
    pipeline = Pipeline([SoftmaxTemperature(), StandardScale()])

    with pytest.raises(RuntimeError, match=r"step 1 \(StandardScale\)"):
        pipeline.transform(_table())


def test_inverse_transform_rejects_non_invertible_step() -> None:
    # MeanImpute is not invertible, so inverse_transform reports its position.
    with pytest.raises(TypeError, match=r"step 0 \(MeanImpute\)"):
        Pipeline([MeanImpute()]).inverse_transform(_table())


def test_target_forward_then_inverse_round_trips() -> None:
    recipe = Recipe(target=[StandardScale()])
    table = _table()

    transformed = recipe.target.fit_transform(table)
    restored = recipe.target.inverse_transform(transformed)

    assert not torch.equal(transformed.numerical, table.numerical)
    assert torch.allclose(restored.numerical, table.numerical, atol=1e-6)


def test_inverse_transform_runs_steps_in_reverse_order() -> None:
    # Power and StandardScale do not commute, so the round trip only
    # reconstructs the input if the inverse applies the steps in reverse.
    table = _table(
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        )
    )

    pipeline = Pipeline([Power(), StandardScale()])
    transformed = pipeline.fit_transform(table)
    restored = pipeline.inverse_transform(transformed)

    assert torch.allclose(restored.numerical, table.numerical, atol=1e-4)


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
