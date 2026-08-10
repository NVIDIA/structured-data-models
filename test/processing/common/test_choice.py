import copy
from typing import Any, cast

import pytest
import torch

import sdm.processing as sp
from sdm import Stype, TableTensor
from sdm.processing import (
    EnsembleProcessor,
    InvertibleMixin,
    Processor,
)
from sdm.tensor import EnsembleTable


class Add(Processor, InvertibleMixin):
    requires_fit = False
    operates_on_stypes = frozenset(Stype)

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical + self.value)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical - self.value)


class AddFittedMemberCount(EnsembleProcessor):
    operates_on_stypes = frozenset(Stype)

    def __init__(self) -> None:
        super().__init__()
        self.value = 0

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.value = ensemble_table.num_members

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return ensemble_table.replace_groups(
            [
                group.replace_blocks(numerical=group.numerical + self.value)
                for group in ensemble_table
            ]
        )


def _table() -> TableTensor:
    return TableTensor.from_tensor(
        torch.arange(64, dtype=torch.float32).view(32, 2),
        columns=("x0", "x1"),
    )


def test_choice_draws_at_fit() -> None:
    choice = sp.Choice(sp.Identity(), sp.Standardize())

    with pytest.raises(RuntimeError, match="no selected option"):
        _ = choice.selected

    choice.fit(_table())

    assert isinstance(choice.selected, sp.Identity | sp.Standardize)


def test_choice_accepts_callable_option() -> None:
    table = _table()
    choice = sp.Choice(
        lambda table: table.replace_blocks(numerical=table.numerical.square())
    )

    output = choice.fit_transform(table)

    assert not choice.selected.requires_fit
    assert torch.equal(output.numerical, table.numerical.square())
    assert repr(choice) == "Choice(\n  Callable(<lambda>),\n)"

    with pytest.raises(AttributeError, match="inverse_transform"):
        choice.inverse_transform(output)


def test_choice_rejects_invalid_option() -> None:
    with pytest.raises(TypeError, match=r"Input must be a"):
        sp.Choice(sp.Identity(), cast(Any, object()))


def test_choice_delegates_fit_transform_and_inverse() -> None:
    choice = sp.Choice(sp.Standardize())
    table = _table()

    transformed = choice.fit_transform(table)
    restored = choice.inverse_transform(transformed)

    assert not torch.equal(transformed.numerical, table.numerical)
    assert torch.allclose(restored.numerical, table.numerical, atol=1e-6)


def test_choice_inverse_requires_invertible_selected() -> None:
    choice = sp.Choice(sp.ImputeMean())
    table = _table()

    choice.fit(table)

    with pytest.raises(AttributeError, match="inverse_transform"):
        choice.inverse_transform(table)


def test_choice_is_reproducible_with_generator() -> None:
    table = _table()

    # Seed 1 draws the second option, whose fit consumes the generator.
    first = sp.Choice(
        sp.Identity(),
        sp.QuantileTransform(n_quantiles=6, subsample=16),
    ).fit_transform(
        table,
        generator=torch.Generator().manual_seed(1),
    )
    second = sp.Choice(
        sp.Identity(),
        sp.QuantileTransform(n_quantiles=6, subsample=16),
    ).fit_transform(
        table,
        generator=torch.Generator().manual_seed(1),
    )

    assert first.equal(second)
    assert not first.equal(table)


def test_choice_repr_shows_all_options() -> None:
    choice = sp.Choice(sp.Identity(), sp.Standardize())

    assert "Identity" in repr(choice)
    assert "Standardize" in repr(choice)


def test_choice_copies_draw_independently_at_fit() -> None:
    template = sp.Choice(
        sp.Identity(),
        sp.QuantileTransform(output_distribution="normal"),
    )
    copies = [copy.deepcopy(template) for _ in range(8)]

    torch.manual_seed(123)
    for choice in copies:
        choice.fit(_table())

    picks = {type(choice.selected).__name__ for choice in copies}
    assert picks == {"Identity", "QuantileTransform"}


def test_choice_round_robin_routes_members() -> None:
    context = _table()
    query = context.replace_blocks(numerical=context.numerical + 100)
    processor = sp.Choice(Add(0), Add(10), method="round_robin")

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
            transformed.table(member_id).numerical,
            context.numerical + offset,
        )
        torch.testing.assert_close(
            query_transformed.table(member_id).numerical,
            query.numerical + offset,
        )
        assert restored.table(member_id).equal(context)


def test_choice_fit_ensemble_fits_selected_options() -> None:
    table = EnsembleTable(_table(), num_members=4)
    fitted = sp.Choice(sp.Standardize(), sp.Identity(), method="round_robin")
    combined = sp.Choice(sp.Standardize(), sp.Identity(), method="round_robin")

    fitted.fit_ensemble(table)
    transformed = fitted.transform_ensemble(table)
    expected = combined.fit_transform_ensemble(table)

    for member_id in range(table.num_members):
        torch.testing.assert_close(
            transformed.table(member_id).numerical,
            expected.table(member_id).numerical,
        )


def test_choice_fits_options_on_selected_members() -> None:
    base = _table()
    tables = tuple(
        base.replace_blocks(numerical=base.numerical + value)
        for value in range(4)
    )
    table = EnsembleTable.from_tables(
        tables=tables,
        member_table_ids=tuple(range(4)),
    )

    output = sp.Choice(
        AddFittedMemberCount(),
        AddFittedMemberCount(),
        method="round_robin",
    ).fit_transform_ensemble(table)

    for member_id, source in enumerate(tables):
        torch.testing.assert_close(
            output.table(member_id).numerical,
            source.numerical + 2,
        )


def test_nested_choice_routes_selected_members_locally() -> None:
    processor = sp.Choice(
        sp.Choice(
            Add(10),
            Add(20),
            Add(30),
            method="round_robin",
        ),
        Add(100),
        method="round_robin",
    )

    output = processor.fit_transform_ensemble(
        EnsembleTable(_table(), num_members=8)
    )

    for member_id in range(8):
        offset = 10 * ((member_id // 2) % 3 + 1) if member_id % 2 == 0 else 100
        torch.testing.assert_close(
            output.table(member_id).numerical,
            _table().numerical + offset,
        )


def test_choice_round_robin_uses_first_option_for_single_table() -> None:
    output = sp.Choice(
        Add(1),
        Add(2),
        method="round_robin",
    ).fit_transform(_table())

    torch.testing.assert_close(output.numerical, _table().numerical + 1)


def test_choice_ensemble_requires_matching_member_count() -> None:
    processor = sp.Choice(Add(0), Add(1), method="round_robin")
    processor.fit_transform_ensemble(EnsembleTable(_table(), num_members=8))

    with pytest.raises(RuntimeError, match="fitted with 8"):
        processor.transform_ensemble(EnsembleTable(_table(), num_members=7))
