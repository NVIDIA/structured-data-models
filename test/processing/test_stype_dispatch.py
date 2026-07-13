import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import (
    Identity,
    MeanImpute,
    StandardScale,
    StypeDispatch,
    ToNumerical,
)


def _mixed_table() -> TableTensor:
    return TableTensor(
        columns={
            "numerical": ("x0", "x1"),
            "categorical": ("kind",),
        },
        numerical=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )


def test_stype_dispatch_routes_and_passes_through_by_default() -> None:
    table = _mixed_table()
    dispatch = StypeDispatch(numerical=StandardScale())

    output = dispatch.fit_transform(table)

    assert output.columns == table.columns
    assert torch.allclose(
        output.numerical.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )
    assert torch.equal(
        output.categorical.as_tensor(),
        table.categorical.as_tensor(),
    )

    restored = dispatch.inverse_transform(output)

    assert restored.columns == table.columns
    assert torch.allclose(restored.numerical, table.numerical)
    assert torch.equal(
        restored.categorical.as_tensor(), table.categorical.as_tensor()
    )


def test_stype_dispatch_inverse_rejects_noninvertible_route() -> None:
    table = _mixed_table()
    dispatch = StypeDispatch(numerical=MeanImpute())

    output = dispatch.fit_transform(table)

    with pytest.raises(TypeError, match=r"numerical.*MeanImpute"):
        dispatch.inverse_transform(output)


def test_stype_dispatch_inverse_rejects_dropped_remainder() -> None:
    table = _mixed_table()
    dispatch = StypeDispatch(
        numerical=StandardScale(),
        remainder="drop",
    )

    output = dispatch.fit_transform(table)

    with pytest.raises(ValueError, match="remainder='drop'"):
        dispatch.inverse_transform(output)


def test_stype_dispatch_rejects_remainder_before_fitting_routes() -> None:
    table = _mixed_table()
    processor = StandardScale()
    dispatch = StypeDispatch(
        numerical=processor,
        remainder="error",
    )

    with pytest.raises(ValueError, match=r"non-empty.*categorical.*no route"):
        dispatch.fit(table)

    with pytest.raises(RuntimeError, match=r"StandardScale.*not fitted"):
        processor.transform(table.select_stypes(Stype.numerical))


def test_stype_dispatch_drops_remainder_and_empty_outputs() -> None:
    output = StypeDispatch(remainder="drop").fit_transform(_mixed_table())

    assert output.size() == (2, 0)
    assert output.columns == {
        Stype.numerical: (),
        Stype.categorical: (),
        Stype.datetime: (),
        Stype.id: (),
    }


def test_stype_dispatch_runs_iterable_routes() -> None:
    dispatch = StypeDispatch(
        numerical=[MeanImpute(), StandardScale()],
        remainder="drop",
    )

    output = dispatch.fit_transform(_mixed_table())

    assert torch.allclose(
        output.numerical.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )


def test_stype_dispatch_uses_route_fitted_state() -> None:
    dispatch = StypeDispatch(
        numerical=StandardScale(),
        remainder="drop",
    )

    with pytest.raises(RuntimeError, match=r"StypeDispatch.*not fitted"):
        dispatch.transform(_mixed_table())

    transformed = dispatch.fit_transform(_mixed_table())
    assert torch.allclose(
        transformed.numerical.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )


def test_stype_dispatch_can_order_routes_for_external_model_contract() -> None:
    dispatch = StypeDispatch(
        numerical=Identity(),
        categorical=ToNumerical(),
        route_order=(Stype.categorical, Stype.numerical),
    )

    output = dispatch.fit_transform(_mixed_table())

    assert output.columns[Stype.numerical] == ("kind", "x0", "x1")
    torch.testing.assert_close(
        output.numerical,
        torch.tensor([[0.0, 1.0, 2.0], [1.0, 3.0, 4.0]]),
    )


def test_stype_dispatch_rejects_duplicate_route_order() -> None:
    with pytest.raises(ValueError, match=r"route_order.*duplicates"):
        StypeDispatch(
            numerical=Identity(),
            route_order=(Stype.numerical, Stype.numerical),
        )
