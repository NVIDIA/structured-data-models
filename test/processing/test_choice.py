import copy
from typing import Any, cast

import pytest
import torch
from sdm import TableTensor
from sdm.processing import (
    Choice,
    Identity,
    MeanImpute,
    Quantile,
    StandardScale,
)


def _table() -> TableTensor:
    return TableTensor.from_tensor(
        torch.arange(64, dtype=torch.float32).view(32, 2),
        columns=("x0", "x1"),
    )


def test_choice_draws_at_fit() -> None:
    choice = Choice(Identity(), StandardScale())

    with pytest.raises(RuntimeError, match="no drawn option"):
        _ = choice.selected

    choice.fit(_table())

    assert choice.selected in list(choice.options)


def test_choice_accepts_callable_option() -> None:
    table = _table()
    choice = Choice(
        lambda table: table.replace_blocks(numerical=table.numerical.square())
    )

    output = choice.fit_transform(table)

    assert not choice.selected.requires_fit
    assert torch.equal(output.numerical, table.numerical.square())
    assert repr(choice) == "Choice(\n  lambda,\n)"

    with pytest.raises(AttributeError, match="inverse_transform"):
        choice.inverse_transform(output)


def test_choice_rejects_invalid_option() -> None:
    with pytest.raises(
        TypeError,
        match=r"Choice option 1.*Processor or callable.*object",
    ):
        Choice(Identity(), cast(Any, object()))


def test_choice_keeps_state_dict_keys_independent_of_the_draw() -> None:
    table = _table()
    torch.manual_seed(0)
    first = Choice(Identity(), StandardScale()).fit(table)
    torch.manual_seed(1)
    second = Choice(Identity(), StandardScale()).fit(table)

    assert set(first.state_dict()) == set(second.state_dict())


def test_choice_delegates_fit_transform_and_inverse() -> None:
    choice = Choice(StandardScale())
    table = _table()

    transformed = choice.fit_transform(table)
    restored = choice.inverse_transform(transformed)

    assert choice.selected._fitted
    assert not torch.equal(transformed.numerical, table.numerical)
    assert torch.allclose(restored.numerical, table.numerical, atol=1e-6)


def test_choice_inverse_requires_invertible_selected() -> None:
    choice = Choice(MeanImpute())
    table = _table()

    choice.fit(table)

    with pytest.raises(AttributeError, match="inverse_transform"):
        choice.inverse_transform(table)


def test_choice_is_reproducible_with_generator() -> None:
    table = _table()

    # Seed 1 draws the second option, whose fit consumes the generator.
    first = Choice(Identity(), Quantile(n_quantiles=6, subsample=16)).fit(
        table,
        generator=torch.Generator().manual_seed(1),
    )
    second = Choice(Identity(), Quantile(n_quantiles=6, subsample=16)).fit(
        table,
        generator=torch.Generator().manual_seed(1),
    )

    assert isinstance(first.selected, Quantile)
    assert type(first.selected) is type(second.selected)
    assert torch.equal(first.selected.quantiles, second.selected.quantiles)


def test_choice_repr_shows_all_options() -> None:
    choice = Choice(Identity(), StandardScale())

    assert "Identity" in repr(choice)
    assert "StandardScale" in repr(choice)


def test_choice_copies_draw_independently_at_fit() -> None:
    template = Choice(
        Identity(),
        Quantile(output_distribution="normal"),
    )
    copies = [copy.deepcopy(template) for _ in range(8)]

    torch.manual_seed(123)
    for choice in copies:
        choice.fit(_table())

    picks = {type(choice.selected).__name__ for choice in copies}
    assert picks == {"Identity", "Quantile"}
