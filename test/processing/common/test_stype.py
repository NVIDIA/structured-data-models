from typing import Any, cast

import pytest
import torch

from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import (
    DropStypes,
    Identity,
    ImputeMean,
    Sequential,
    ShuffleCategories,
    Standardize,
    StypeDispatch,
)


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
    dispatch = StypeDispatch(numerical=Standardize())

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
    dispatch = StypeDispatch(
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

    with pytest.raises(TypeError, match="non-invertible"):
        dispatch.inverse_transform(output)


def test_stype_dispatch_passes_generator_to_routes() -> None:
    table = TableTensor(
        columns={"categorical": ("kind",)},
        categorical=CategoricalTensor(
            code=torch.arange(6, dtype=torch.int32).unsqueeze(1),
            categories=(StringTensor.from_list(list("abcdef")),),
        ),
    )

    first = ShuffleCategories(method="random")
    StypeDispatch(categorical=first).fit(
        table,
        generator=torch.Generator().manual_seed(0),
    )
    second = ShuffleCategories(method="random")
    StypeDispatch(categorical=second).fit(
        table,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.equal(first.permutations, second.permutations)


def test_stype_dispatch_inverse_rejects_noninvertible_route() -> None:
    table = _mixed_table()
    dispatch = StypeDispatch(numerical=ImputeMean())

    output = dispatch.fit_transform(table)

    with pytest.raises(TypeError, match=r"numerical.*ImputeMean"):
        dispatch.inverse_transform(output)


def test_stype_dispatch_rejects_drop_remainder() -> None:
    with pytest.raises(ValueError, match=r"passthrough.*error"):
        StypeDispatch(remainder=cast(Any, "drop"))


def test_stype_dispatch_rejects_remainder_before_fitting_routes() -> None:
    table = _mixed_table()
    processor = Standardize()
    dispatch = StypeDispatch(
        numerical=processor,
        remainder="error",
    )

    with pytest.raises(
        ValueError, match=r"StypeDispatch.*cannot preserve.*categorical"
    ):
        dispatch.fit(table)

    with pytest.raises(RuntimeError, match=r"Standardize.*not fitted"):
        processor.transform(table.select_stypes(Stype.numerical))


def test_drop_stypes_removes_configured_stypes() -> None:
    output = DropStypes(Stype.numerical).fit_transform(_mixed_table())

    assert output.columns[Stype.numerical] == ()
    assert output.columns[Stype.categorical] == ("kind",)
    assert torch.equal(
        output.categorical.code, _mixed_table().categorical.code
    )


def test_drop_stypes_noops_without_matching_stypes() -> None:
    table = _mixed_table()

    assert DropStypes(Stype.id).fit_transform(table) is table


def test_stype_dispatch_runs_iterable_routes() -> None:
    table = _mixed_table().replace_blocks(
        numerical=torch.tensor([[-3.0, 2.0], [1.0, 4.0]])
    )
    dispatch = Sequential(
        StypeDispatch(
            numerical=[
                lambda table: table.replace_blocks(
                    numerical=table.numerical.square()
                ),
                ImputeMean(),
                Standardize(),
            ],
        ),
        DropStypes(Stype.categorical),
    )
    expected = Standardize().fit_transform(
        table.select_stypes(Stype.numerical).replace_blocks(
            numerical=table.numerical.square()
        )
    )

    output = dispatch.fit_transform(table)

    torch.testing.assert_close(output.numerical, expected.numerical)


def test_stype_dispatch_routes_text() -> None:
    table = TableTensor(
        columns={"text": ("review",)},
        text=StringTensor.from_list([["good"], ["bad"]]),
    )
    dispatch = StypeDispatch(text=Identity())

    output = dispatch.fit_transform(table)

    assert output.columns[Stype.text] == ("review",)
    assert output.text.equal(table.text)


def test_stype_dispatch_uses_route_fitted_state() -> None:
    dispatch = StypeDispatch(numerical=Standardize())

    with pytest.raises(RuntimeError, match=r"StypeDispatch.*not fitted"):
        dispatch.transform(_mixed_table())

    transformed = dispatch.fit_transform(_mixed_table())
    assert torch.allclose(
        transformed.numerical.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )
