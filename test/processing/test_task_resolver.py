import copy
from unittest.mock import patch

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import (
    InvertibleMixin,
    Processor,
)
from sdm.testing import withCUDA


def _categorical_target(device: torch.device | None = None) -> TableTensor:
    return TableTensor(
        columns={"categorical": ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[0], [1]],
                dtype=torch.int32,
                device=device,
            ),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )


def _numerical_target(device: torch.device | None = None) -> TableTensor:
    return TableTensor.from_tensor(
        torch.arange(2, dtype=torch.float, device=device).unsqueeze(-1),
        columns=("target",),
    )


def _output(device: torch.device | None = None) -> TableTensor:
    return TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [2.0, 1.0]], device=device),
        columns=("a", "b"),
    )


def _recipe(*, target: Processor | None = None) -> sp.Recipe:
    return sp.Recipe(
        target=target,
        output=[
            sp.TaskDispatch(
                classification=sp.Softmax(),
                regression=sp.Identity(),
            )
        ],
    )


@withCUDA
def test_task_resolver_uses_final_target_type_once(
    device: torch.device,
) -> None:
    target_processor = sp.Identity()
    recipe = _recipe(target=target_processor)
    categorical_target = _categorical_target(device)
    output = _output(device)

    with patch.object(
        target_processor,
        "_transform",
        wraps=target_processor._transform,
    ) as transform:
        transformed_target = recipe.target.fit_transform(categorical_target)

    assert transformed_target.equal(categorical_target)
    assert transform.call_count == 1
    assert torch.allclose(
        recipe.output.transform(output).numerical.sum(dim=-1),
        torch.ones(2, device=device),
    )

    recipe.target.fit(_numerical_target(device))
    assert recipe.output.transform(output).equal(output)

    converted = sp.Recipe(
        target=[sp.ToNumerical()],
        output=[sp.TaskDispatch(regression=sp.Identity())],
    )
    transformed_target = converted.target.fit_transform(categorical_target)

    assert transformed_target.numerical.size(-1) == 1
    assert converted.output.transform(output).equal(output)

    scaled = sp.Recipe(
        target=[sp.Standardize()],
        output=[sp.TaskDispatch(regression=sp.Identity())],
    )
    numerical_target = _numerical_target(device)
    transformed_target = scaled.target.fit_transform(numerical_target)

    assert isinstance(scaled.target, InvertibleMixin)
    restored_target = scaled.target.inverse_transform(transformed_target)
    torch.testing.assert_close(
        restored_target.numerical,
        numerical_target.numerical,
    )


def test_task_resolver_clears_failures_and_validates_placement() -> None:
    output = _output()
    recipe = sp.Recipe(output=[sp.TaskDispatch(regression=sp.Identity())])
    assert "_TaskResolver" not in repr(recipe)

    recipe.target.fit(_numerical_target())
    assert recipe.output.transform(output).equal(output)

    recipe.target.fit(_categorical_target())
    assert recipe.output.transform(output).equal(output)

    with pytest.raises(ValueError, match=r"not supported.*Recipe.target"):
        sp.Recipe(target=[sp.TaskDispatch(regression=sp.Identity())])


def test_task_resolver_copies_recipes_independently() -> None:
    template = _recipe()
    assert not any(
        isinstance(module, sp.TaskDispatch)
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
        torch.ones(2),
    )
    assert regression.output.transform(output).equal(output)
