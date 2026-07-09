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


def test_choice_draws_at_fit() -> None:
    choice = Choice([Identity(), StandardScale()])

    with pytest.raises(RuntimeError, match="not fitted"):
        choice.transform(_table())
    with pytest.raises(RuntimeError, match="no drawn option"):
        _ = choice.selected

    choice.fit(_table())

    assert choice.selected in list(choice.options)


def test_choice_keeps_state_dict_keys_independent_of_the_draw() -> None:
    table = _table()
    torch.manual_seed(0)
    first = Choice([Identity(), StandardScale()]).fit(table)
    torch.manual_seed(1)
    second = Choice([Identity(), StandardScale()]).fit(table)

    assert set(first.state_dict()) == set(second.state_dict())


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


def test_choice_repr_shows_options_then_the_drawn_option() -> None:
    choice = Choice([Identity(), StandardScale()])

    assert "Identity" in repr(choice)
    assert "StandardScale" in repr(choice)

    torch.manual_seed(0)
    choice.fit(_table())

    drawn = type(choice.selected).__name__
    assert repr(choice).count("(") == 2
    assert drawn in repr(choice)


def test_recipe_members_draw_independent_options_at_fit() -> None:
    template = _make_recipe()
    members = [copy.deepcopy(template) for _ in range(8)]

    torch.manual_seed(123)
    for member in members:
        member.features.fit(_table())

    picks = {
        type(_choice_step(member).selected).__name__ for member in members
    }
    assert picks == {"Identity", "Quantile"}


def test_fitting_one_member_leaves_others_unfitted() -> None:
    first = _make_recipe()
    second = _make_recipe()

    torch.manual_seed(0)
    first.features.fit(_table())

    assert first.features._fitted
    assert not second.features._fitted
    assert not any(option._fitted for option in _choice_step(second).options)


def test_deepcopy_of_fitted_member_keeps_the_draw() -> None:
    member = _make_recipe()
    torch.manual_seed(0)
    member.features.fit(_table())

    duplicate = copy.deepcopy(member)

    original_choice = _choice_step(member)
    duplicate_choice = _choice_step(duplicate)
    assert type(duplicate_choice.selected) is type(original_choice.selected)
    assert duplicate_choice.selected is not original_choice.selected
