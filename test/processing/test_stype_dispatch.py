import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import (
    Identity,
    InvertibleMixin,
    MeanImpute,
    Processor,
    StandardScale,
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
            data=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )


class _BinaryOneHot(Processor, InvertibleMixin):
    supported_stypes = frozenset({Stype.numerical, Stype.categorical})
    requires_fit = False

    def _transform(self, input: TableTensor) -> TableTensor:
        numerical = torch.nn.functional.one_hot(
            input.categorical.as_tensor().long().squeeze(-1),
            num_classes=2,
        ).to(dtype=torch.get_default_dtype())
        return TableTensor.from_tensor(
            numerical,
            columns=("kind_a", "kind_b"),
        )

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        categorical = CategoricalTensor(
            data=input.numerical.argmax(dim=-1, keepdim=True).to(torch.int32),
            categories=(StringTensor.from_list(["a", "b"]),),
        )
        return TableTensor(
            columns={Stype.categorical: ("kind",)},
            categorical=categorical,
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


def test_stype_dispatch_inverse_routes_width_changing_outputs() -> None:
    table = _mixed_table()
    dispatch = StypeDispatch(
        numerical=Identity(),
        categorical=_BinaryOneHot(),
    )

    output = dispatch.fit_transform(table)

    assert output.columns[Stype.numerical] == (
        "x0",
        "x1",
        "kind_a",
        "kind_b",
    )
    prediction = TableTensor.from_tensor(
        output.numerical,
        columns=(
            "prediction_0",
            "prediction_1",
            "prediction_2",
            "prediction_3",
        ),
    )

    restored = dispatch.inverse_transform(prediction)

    assert restored.columns[Stype.numerical] == (
        "prediction_0",
        "prediction_1",
    )
    assert restored.columns[Stype.categorical] == ("kind",)
    assert torch.allclose(restored.numerical, table.numerical)
    assert torch.equal(
        restored.categorical.as_tensor(),
        table.categorical.as_tensor(),
    )


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


# TODO: Cover shape-changing routes once ConstantFilter is available.


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
