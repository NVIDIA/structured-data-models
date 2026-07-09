import copy
from unittest.mock import patch

import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import (
    Identity,
    InvertibleMixin,
    Processor,
    Recipe,
    Sequential,
    SoftmaxTemperature,
    StandardScale,
    TaskDispatch,
    ToNumerical,
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


def _numerical_target(device: torch.device | None = None) -> TableTensor:
    return TableTensor.from_tensor(
        torch.arange(4, dtype=torch.float, device=device).unsqueeze(-1),
        columns=("target",),
    )


def _output(device: torch.device | None = None) -> TableTensor:
    return TableTensor.from_tensor(
        torch.tensor(
            [[1.0, 2.0], [2.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            device=device,
        ),
        columns=("a", "b"),
    )


def _recipe(*, target: Processor | None = None) -> Recipe:
    return Recipe(
        target=target,
        output=[
            TaskDispatch(
                classification=SoftmaxTemperature(),
                regression=Identity(),
            )
        ],
    )


@withCUDA
def test_recipe_target_resolves_and_refits_task_dispatch(
    device: torch.device,
) -> None:
    target_processor = Identity()
    recipe = _recipe(target=target_processor)
    categorical_target = _categorical_target(device)
    output = _output(device)

    with patch.object(
        target_processor,
        "_transform",
        wraps=target_processor._transform,
    ) as transform:
        transformed_target = recipe.target.fit_transform(categorical_target)

    assert transformed_target is categorical_target
    assert transform.call_count == 1
    classification_output = recipe.output.transform(output)
    assert torch.allclose(
        classification_output.numerical.sum(dim=-1),
        torch.ones(4, device=device),
    )

    recipe.target.fit(_numerical_target(device))

    assert recipe.output.transform(output) is output

    scaled_recipe = _recipe(target=StandardScale())
    numerical_target = _numerical_target(device)
    transformed_target = scaled_recipe.target.fit_transform(numerical_target)

    assert isinstance(scaled_recipe.target, InvertibleMixin)
    restored_target = scaled_recipe.target.inverse_transform(
        transformed_target
    )
    torch.testing.assert_close(
        restored_target.numerical,
        numerical_target.numerical,
    )


def test_task_dispatch_uses_final_transformed_target_type() -> None:
    recipe = Recipe(
        target=[ToNumerical()],
        output=[
            TaskDispatch(
                classification=SoftmaxTemperature(),
                regression=Identity(),
            )
        ],
    )
    output = _output()

    transformed_target = recipe.target.fit_transform(_categorical_target())

    assert transformed_target.numerical.size(-1) == 1
    assert recipe.output.transform(output) is output


def test_recipe_copies_and_restores_task_dispatch_state() -> None:
    template = _recipe()
    assert not any(
        isinstance(module, TaskDispatch)
        for module in template.target.modules()
    )
    classification = copy.deepcopy(template)
    regression = copy.deepcopy(template)
    output = _output()

    classification.target.fit(_categorical_target())
    regression.target.fit(_numerical_target())

    with pytest.raises(RuntimeError, match=r"recipe\.target\.fit"):
        template.output.transform(output)
    assert torch.allclose(
        classification.output.transform(output).numerical.sum(dim=-1),
        torch.ones(4),
    )
    assert regression.output.transform(output) is output

    restored = copy.deepcopy(template)
    restored.output.load_state_dict(classification.output.state_dict())

    assert torch.allclose(
        restored.output.transform(output).numerical,
        classification.output.transform(output).numerical,
    )


def test_task_dispatch_rejects_invalid_configuration_and_placement() -> None:
    with pytest.raises(ValueError, match="at least one route"):
        TaskDispatch()

    with pytest.raises(ValueError, match=r"regression.*requires fit"):
        TaskDispatch(regression=StandardScale())

    with pytest.raises(ValueError, match=r"only supported.*Recipe.output"):
        Recipe(features=[TaskDispatch(regression=Identity())])
    with pytest.raises(ValueError, match=r"only supported.*Recipe.output"):
        Recipe(target=[TaskDispatch(regression=Identity())])
    with pytest.raises(ValueError, match=r"direct step.*Recipe.output"):
        Recipe(
            output=[
                Sequential(
                    TaskDispatch(regression=Identity()),
                )
            ]
        )


def test_task_dispatch_rejects_unresolved_or_invalid_tasks() -> None:
    output = _output()
    recipe = Recipe(output=[TaskDispatch(regression=Identity())])

    with pytest.raises(RuntimeError, match=r"recipe\.target\.fit"):
        recipe.output.transform(output)

    recipe.target.fit(_numerical_target())
    assert recipe.output.transform(output) is output

    with pytest.raises(ValueError, match="no 'classification' route"):
        recipe.target.fit(_categorical_target())
    with pytest.raises(RuntimeError, match=r"recipe\.target\.fit"):
        recipe.output.transform(output)

    ambiguous_target = TableTensor.from_tensor(
        torch.ones(4, 2),
        columns=("y0", "y1"),
    )
    with pytest.raises(ValueError, match=r"exactly one.*got 2"):
        recipe.target.fit(ambiguous_target)
