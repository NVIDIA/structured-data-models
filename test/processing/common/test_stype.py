import pytest
import torch

import sdm.processing as sp
from sdm import (
    CategoricalTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.tensor import EnsembleTable


def _mixed_table() -> TableTensor:
    return TableTensor(
        columns={
            "numerical": ("x0", "x1"),
            "categorical": ("kind",),
        },
        numerical=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )


def test_stype_dispatch_routes_and_passes_through_by_default() -> None:
    table = _mixed_table()
    dispatch = sp.StypeDispatch(numerical=sp.Standardize())

    output = dispatch.fit_transform(table)

    assert output.columns == table.columns
    assert torch.allclose(
        output.numerical.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )
    assert torch.equal(
        output.categorical.code,
        table.categorical.code,
    )

    restored = dispatch.inverse_transform(output)

    assert restored.columns == table.columns
    assert torch.allclose(restored.numerical, table.numerical)
    assert torch.equal(restored.categorical.code, table.categorical.code)


def test_stype_dispatch_accepts_callable_route() -> None:
    table = _mixed_table()
    dispatch = sp.StypeDispatch(
        numerical=lambda table: table.replace_blocks(
            numerical=table.numerical.square()
        )
    )

    output = dispatch.transform(table)

    assert not dispatch.requires_fit
    assert output.columns == table.columns
    assert torch.equal(
        output.numerical,
        table.numerical.square(),
    )
    assert torch.equal(
        output.categorical.code,
        table.categorical.code,
    )

    with pytest.raises(AttributeError, match="inverse_transform"):
        dispatch.inverse_transform(output)


def test_stype_dispatch_passes_generator_to_routes() -> None:
    table = TableTensor(
        columns={"categorical": ("kind",)},
        categorical=CategoricalTensor(
            code=torch.arange(6, dtype=torch.int32).unsqueeze(1),
            categories=(StringTensor.from_list(list("abcdef")),),
        ),
    )

    first = sp.StypeDispatch(categorical=sp.ShuffleCategories(method="random"))
    first_output = first.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )
    second = sp.StypeDispatch(
        categorical=sp.ShuffleCategories(method="random")
    )
    second_output = second.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )

    assert first_output.equal(second_output)


def test_stype_dispatch_inverse_rejects_noninvertible_route() -> None:
    table = _mixed_table()
    dispatch = sp.StypeDispatch(numerical=sp.ImputeMean())

    output = dispatch.fit_transform(table)

    with pytest.raises(AttributeError, match="inverse_transform"):
        dispatch.inverse_transform(output)


def test_stype_dispatch_runs_iterable_routes() -> None:
    table = _mixed_table().replace_blocks(
        numerical=torch.tensor([[-3.0, 2.0], [1.0, 4.0]])
    )
    dispatch = sp.StypeDispatch(
        numerical=[
            lambda table: table.replace_blocks(
                numerical=table.numerical.square()
            ),
            sp.ImputeMean(),
            sp.Standardize(),
        ],
    )
    expected = sp.Standardize().fit_transform(
        table.select_stypes(Stype.numerical).replace_blocks(
            numerical=table.numerical.square()
        )
    )

    output = dispatch.fit_transform(table)

    assert output.columns == table.columns
    torch.testing.assert_close(output.numerical, expected.numerical)
    assert output.categorical.equal(table.categorical)


def test_stype_dispatch_routes_text() -> None:
    table = TableTensor(
        columns={"text": ("review",)},
        text=StringTensor.from_list([["good"], ["bad"]]),
    )
    dispatch = sp.StypeDispatch(text=sp.Identity())

    output = dispatch.fit_transform(table)

    assert output.columns[Stype.text] == ("review",)
    assert output.text.equal(table.text)


def test_stype_dispatch_uses_route_fitted_state() -> None:
    dispatch = sp.StypeDispatch(numerical=sp.Standardize())

    with pytest.raises(RuntimeError, match=r"StypeDispatch.*not fitted"):
        dispatch.transform(_mixed_table())

    transformed = dispatch.fit_transform(_mixed_table())
    assert torch.allclose(
        transformed.numerical.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )


def test_stype_dispatch_ensemble_routes_members_and_preserves_order() -> None:
    first = _mixed_table()
    second = _mixed_table().replace_blocks(
        numerical=torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    )
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(1, 0, 1),
    )
    processor = sp.StypeDispatch(
        numerical=lambda value: value.replace_blocks(
            numerical=value.numerical.square()
        )
    )

    output = processor.transform_ensemble(table)

    for member_id, source in enumerate((second, first, second)):
        result = output.table(member_id)
        assert torch.equal(result.numerical, source.numerical.square())
        assert result.categorical.equal(source.categorical)


def test_stype_dispatch_ensemble_fits_routes_per_group() -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[1.0], [3.0]]),
        columns=("first",),
    )
    second = TableTensor.from_tensor(
        torch.tensor([[10.0], [14.0]]),
        columns=("second",),
    )
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 0),
    )
    processor = sp.StypeDispatch(numerical=sp.Standardize(with_std=False))
    combined = sp.StypeDispatch(numerical=sp.Standardize(with_std=False))

    processor.fit_ensemble(table)
    transformed = processor.transform_ensemble(table)
    expected = combined.fit_transform_ensemble(table)

    for member_id in range(table.num_members):
        result = transformed.table(member_id)
        assert result.equal(expected.table(member_id))
        assert torch.allclose(
            result.numerical.mean(dim=-2),
            torch.zeros(1),
        )
