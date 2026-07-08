import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import (
    Identity,
    Recipe,
    SoftmaxTemperature,
    StandardScale,
    TaskDispatch,
)
from sdm.testing import withCUDA


def _categorical_target(device: torch.device | None = None) -> TableTensor:
    return TableTensor(
        columns={"categorical": ("target",)},
        categorical=CategoricalTensor(
            data=torch.tensor(
                [[0], [1], [0], [1]],
                dtype=torch.int32,
                device=device,
            ),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )


@withCUDA
def test_recipe_resolves_refits_and_restores_task_dispatch(
    device: torch.device,
) -> None:
    features = TableTensor.from_tensor(
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            device=device,
        ),
        columns=("x0", "x1"),
    )
    output = TableTensor.from_tensor(
        torch.tensor(
            [[1.0, 2.0], [2.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            device=device,
        ),
        columns=("a", "b"),
    )
    recipe = Recipe(
        features=[StandardScale()],
        output=[
            TaskDispatch(
                classification=SoftmaxTemperature(),
                regression=Identity(),
            )
        ],
    )

    transformed_features, _ = recipe.fit_transform(
        features,
        _categorical_target(device),
    )
    classification_output = recipe.output.transform(output)

    assert torch.allclose(
        transformed_features.numerical.mean(dim=0),
        torch.zeros(2, device=device),
        atol=1e-6,
    )
    assert torch.allclose(
        classification_output.numerical.sum(dim=-1),
        torch.ones(4, device=device),
    )

    restored = Recipe(
        output=[
            TaskDispatch(
                classification=SoftmaxTemperature(),
                regression=Identity(),
            )
        ],
    )
    restored.output.load_state_dict(recipe.output.state_dict())
    torch.testing.assert_close(
        restored.output.transform(output).numerical,
        classification_output.numerical,
    )

    numerical_target = TableTensor.from_tensor(
        torch.arange(4, dtype=torch.float, device=device).unsqueeze(-1),
        columns=("target",),
    )
    recipe.fit_transform(features, numerical_target)

    assert recipe.output.transform(output) is output


def test_task_dispatch_rejects_invalid_configuration_and_placement() -> None:
    with pytest.raises(ValueError, match="at least one route"):
        TaskDispatch()

    with pytest.raises(ValueError, match=r"regression.*requires fit"):
        TaskDispatch(regression=StandardScale())

    with pytest.raises(ValueError, match=r"only supported.*Recipe.output"):
        Recipe(features=[TaskDispatch(regression=Identity())])


def test_task_dispatch_rejects_unresolved_or_invalid_tasks() -> None:
    output = TableTensor.from_tensor(
        torch.ones(4, 1),
        columns=("prediction",),
    )
    dispatch = TaskDispatch(regression=Identity())

    with pytest.raises(RuntimeError, match=r"Recipe.fit_transform"):
        dispatch.transform(output)

    recipe = Recipe(output=[dispatch])
    recipe.fit_transform(output, output)
    with pytest.raises(ValueError, match="no 'classification' route"):
        recipe.fit_transform(output, _categorical_target())
    with pytest.raises(RuntimeError, match=r"Recipe.fit_transform"):
        recipe.output.transform(output)

    ambiguous_target = TableTensor.from_tensor(
        torch.ones(4, 2),
        columns=("y0", "y1"),
    )
    recipe = Recipe(
        output=[
            TaskDispatch(
                classification=Identity(),
                regression=Identity(),
            )
        ],
    )
    with pytest.raises(ValueError, match=r"exactly one.*got 2"):
        recipe.fit_transform(output, ambiguous_target)
