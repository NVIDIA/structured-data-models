import copy
from typing import Any, cast

import pytest
import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import (
    Choice,
    Identity,
    ImputeMean,
    InvertibleMixin,
    Processor,
    QuantileTransform,
    Standardize,
)


class Add(Processor, InvertibleMixin):
    requires_fit = False
    supported_stypes = frozenset(Stype)

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical + self.value)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical - self.value)


def _table() -> TableTensor:
    return TableTensor.from_tensor(
        torch.arange(64, dtype=torch.float32).view(32, 2),
        columns=("x0", "x1"),
    )


def test_choice_draws_at_fit() -> None:
    choice = Choice(Identity(), Standardize())

    with pytest.raises(RuntimeError, match="no drawn option"):
        _ = choice.selected

    choice.fit(_table())

    assert isinstance(choice.selected, Identity | Standardize)


def test_choice_accepts_callable_option() -> None:
    table = _table()
    choice = Choice(
        lambda table: table.replace_blocks(numerical=table.numerical.square())
    )

    output = choice.fit_transform(table)

    assert not choice.selected.requires_fit
    assert torch.equal(output.numerical, table.numerical.square())
    assert repr(choice) == "Choice(\n  Callable(<lambda>),\n)"

    with pytest.raises(TypeError, match="not invertible"):
        choice.inverse_transform(output)


def test_choice_rejects_invalid_option() -> None:
    with pytest.raises(TypeError, match=r"Input must be a"):
        Choice(Identity(), cast(Any, object()))


def test_choice_delegates_fit_transform_and_inverse() -> None:
    choice = Choice(Standardize())
    table = _table()

    transformed = choice.fit_transform(table)
    restored = choice.inverse_transform(transformed)

    assert choice.selected._fitted
    assert not torch.equal(transformed.numerical, table.numerical)
    assert torch.allclose(restored.numerical, table.numerical, atol=1e-6)


def test_choice_inverse_requires_invertible_selected() -> None:
    choice = Choice(ImputeMean())
    table = _table()

    choice.fit(table)

    with pytest.raises(TypeError, match="not invertible"):
        choice.inverse_transform(table)


def test_choice_is_reproducible_with_generator() -> None:
    table = _table()

    # Seed 1 draws the second option, whose fit consumes the generator.
    first = Choice(
        Identity(),
        QuantileTransform(n_quantiles=6, subsample=16),
    ).fit(
        table,
        generator=torch.Generator().manual_seed(1),
    )
    second = Choice(
        Identity(),
        QuantileTransform(n_quantiles=6, subsample=16),
    ).fit(
        table,
        generator=torch.Generator().manual_seed(1),
    )

    assert isinstance(first.selected, QuantileTransform)
    assert type(first.selected) is type(second.selected)
    assert torch.equal(first.selected.quantiles, second.selected.quantiles)


def test_choice_repr_shows_all_options() -> None:
    choice = Choice(Identity(), Standardize())

    assert "Identity" in repr(choice)
    assert "Standardize" in repr(choice)


def test_choice_copies_draw_independently_at_fit() -> None:
    template = Choice(
        Identity(),
        QuantileTransform(output_distribution="normal"),
    )
    copies = [copy.deepcopy(template) for _ in range(8)]

    torch.manual_seed(123)
    for choice in copies:
        choice.fit(_table())

    picks = {type(choice.selected).__name__ for choice in copies}
    assert picks == {"Identity", "QuantileTransform"}


def test_choice_round_robin_routes_eight_members_and_reuses_selection() -> (
    None
):
    context = _table()
    query = context.replace_blocks(numerical=context.numerical + 100)
    processor = Choice(Add(0), Add(10), selection="round_robin")

    transformed = processor.fit_transform_ensemble(
        EnsembleTable(context, num_members=8)
    )
    query_transformed = processor.transform_ensemble(
        EnsembleTable(query, num_members=8)
    )
    restored = processor.inverse_transform_ensemble(transformed)

    for member_id in range(8):
        offset = 10 * (member_id % 2)
        torch.testing.assert_close(
            transformed.representation(member_id).numerical,
            context.numerical + offset,
        )
        torch.testing.assert_close(
            query_transformed.representation(member_id).numerical,
            query.numerical + offset,
        )
        assert restored.representation(member_id).equal(context)


def test_choice_fit_ensemble_fits_selected_options() -> None:
    table = EnsembleTable(_table(), num_members=4)
    fitted = Choice(Standardize(), Identity(), selection="round_robin")
    combined = Choice(Standardize(), Identity(), selection="round_robin")

    fitted.fit_ensemble(table)
    transformed = fitted.transform_ensemble(table)
    expected = combined.fit_transform_ensemble(table)

    for member_id in range(table.num_members):
        torch.testing.assert_close(
            transformed.table(member_id).numerical,
            expected.table(member_id).numerical,
        )


def test_nested_choice_uses_stable_member_positions() -> None:
    processor = Choice(
        Choice(
            Add(10),
            Add(20),
            Add(30),
            selection="round_robin",
        ),
        Add(100),
        selection="round_robin",
    )

    output = processor.fit_transform_ensemble(
        EnsembleTable(_table(), num_members=8)
    )

    for member_id in range(8):
        offset = 10 * (member_id % 3 + 1) if member_id % 2 == 0 else 100
        torch.testing.assert_close(
            output.representation(member_id).numerical,
            _table().numerical + offset,
        )


def test_choice_validates_selection_and_scalar_round_robin() -> None:
    with pytest.raises(ValueError, match="at least one option"):
        Choice()
    with pytest.raises(ValueError, match="round_robin"):
        Choice(Identity(), selection=cast(Any, "unknown"))

    output = Choice(
        Add(1),
        Add(2),
        selection="round_robin",
    ).fit_transform(_table())

    torch.testing.assert_close(output.numerical, _table().numerical + 1)


def test_choice_random_ensemble_is_reproducible() -> None:
    table = EnsembleTable(_table(), num_members=8)
    first = Choice(Add(0), Add(1)).fit_transform_ensemble(
        table,
        generator=torch.Generator().manual_seed(7),
    )
    second = Choice(Add(0), Add(1)).fit_transform_ensemble(
        table,
        generator=torch.Generator().manual_seed(7),
    )

    for member_id in range(table.num_members):
        assert first.representation(member_id).equal(
            second.representation(member_id)
        )


def test_choice_ensemble_requires_matching_member_count() -> None:
    processor = Choice(Add(0), Add(1), selection="round_robin")
    processor.fit_transform_ensemble(EnsembleTable(_table(), num_members=8))

    with pytest.raises(RuntimeError, match="same number"):
        processor.transform_ensemble(EnsembleTable(_table(), num_members=7))


def test_choice_ensemble_inverse_rejects_non_invertible_option() -> None:
    processor = Choice(ImputeMean(), selection="round_robin")
    transformed = processor.fit_transform_ensemble(
        EnsembleTable(_table(), num_members=2)
    )

    with pytest.raises(TypeError, match="not invertible"):
        processor.inverse_transform_ensemble(transformed)
