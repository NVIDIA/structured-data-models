import copy

import pytest
import torch
from sdm import TableTensor
from sdm.processing import (
    Choice,
    Identity,
    MeanImpute,
    Quantile,
    Recipe,
    Sequential,
    StandardScale,
)


def _choice_step(recipe: Recipe) -> Choice:
    features = recipe.features
    assert isinstance(features, Sequential)
    choice = features.steps[1]
    assert isinstance(choice, Choice)
    return choice


def _table(seed: int = 0) -> TableTensor:
    generator = torch.Generator().manual_seed(seed)
    return TableTensor.from_tensor(
        torch.randn(32, 2, generator=generator),
        columns=("x0", "x1"),
    )


def _make_recipe() -> Recipe:
    return Recipe(
        features=[
            MeanImpute(),
            Choice([Identity(), Quantile(output_distribution="normal")]),
        ],
    )


def test_choice_rejects_empty_options() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Choice([])


def test_choice_registers_only_the_selected_option() -> None:
    choice = Choice([Identity(), StandardScale()])

    children = list(choice.children())

    assert len(children) == 1
    assert children[0] is choice.selected


def test_choice_mirrors_requires_fit_of_selected() -> None:
    stateless = Choice([Identity()])
    stateful = Choice([StandardScale()])

    assert stateless.requires_fit is False
    assert stateful.requires_fit is True
    with pytest.raises(RuntimeError, match="not fitted"):
        stateful.transform(_table())


def test_choice_delegates_fit_transform_and_inverse() -> None:
    choice = Choice([StandardScale()])
    table = _table()

    transformed = choice.fit_transform(table)
    restored = choice.inverse_transform(transformed)

    assert choice.selected._fitted
    assert not torch.equal(transformed.numerical, table.numerical)
    assert torch.allclose(restored.numerical, table.numerical, atol=1e-6)


def test_choice_inverse_requires_invertible_selected() -> None:
    choice = Choice([MeanImpute()])
    table = _table()

    choice.fit(table)

    with pytest.raises(AttributeError, match="inverse_transform"):
        choice.inverse_transform(table)


def test_choice_repr_shows_the_selected_option() -> None:
    choice = Choice([StandardScale()])

    assert "Choice(" in repr(choice)
    assert "StandardScale" in repr(choice)


def test_recipe_members_select_independent_options() -> None:
    torch.manual_seed(123)
    members = [_make_recipe() for _ in range(8)]

    picks = {
        type(_choice_step(member).selected).__name__ for member in members
    }

    assert picks == {"Identity", "Quantile"}
    for member in members:
        member.features.fit_transform(_table())


def test_fitting_one_member_leaves_others_unfitted() -> None:
    torch.manual_seed(0)
    first = _make_recipe()
    second = _make_recipe()

    first.features.fit(_table())

    assert first.features._fitted
    assert not second.features._fitted
    assert not _choice_step(second).selected._fitted


def test_deepcopy_keeps_selection_but_not_identity() -> None:
    torch.manual_seed(0)
    member = _make_recipe()

    duplicate = copy.deepcopy(member)

    original_choice = _choice_step(member)
    duplicate_choice = _choice_step(duplicate)
    assert type(duplicate_choice.selected) is type(original_choice.selected)
    assert duplicate_choice.selected is not original_choice.selected
