import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import (
    Identity,
    MeanImpute,
    Processor,
    Sequential,
    StandardScale,
    StypeDispatch,
)


class RecordingFit(Processor):
    supported_stypes = frozenset({Stype.numerical})

    def __init__(self) -> None:
        super().__init__()
        self.fit_called = False

    def _fit(self, input: TableTensor) -> None:
        self.fit_called = True

    def _transform(self, input: TableTensor) -> TableTensor:
        return input


class KeepFirstNumerical(Processor):
    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, input: TableTensor) -> TableTensor:
        return TableTensor.from_tensor(
            input.numerical[..., :1],
            columns=("first",),
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
    dispatch = StypeDispatch({"numerical": StandardScale()})

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


def test_stype_dispatch_rejects_remainder_when_requested() -> None:
    dispatch = StypeDispatch(
        {"numerical": Identity()},
        remainder="error",
    )

    with pytest.raises(ValueError, match="categorical"):
        dispatch.fit_transform(_mixed_table())


def test_stype_dispatch_remainder_error_does_not_fit_routes() -> None:
    processor = RecordingFit()
    dispatch = StypeDispatch(
        {"numerical": processor},
        remainder="error",
    )

    with pytest.raises(ValueError, match="categorical"):
        dispatch.fit(_mixed_table())

    assert not processor.fit_called


def test_stype_dispatch_drops_remainder_and_empty_outputs() -> None:
    output = StypeDispatch({}, remainder="drop").fit_transform(_mixed_table())

    assert output.size() == (2, 0)
    assert output.columns == {
        Stype.numerical: (),
        Stype.categorical: (),
        Stype.datetime: (),
        Stype.id: (),
    }


def test_stype_dispatch_normalizes_iterable_routes() -> None:
    dispatch = StypeDispatch(
        {"numerical": [MeanImpute(), StandardScale()]},
        remainder="drop",
    )

    output = dispatch.fit_transform(_mixed_table())

    assert isinstance(dispatch.processors["numerical"], Sequential)
    assert torch.allclose(
        output.numerical.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )


def test_stype_dispatch_allows_shape_changing_routes() -> None:
    dispatch = StypeDispatch(
        {"numerical": KeepFirstNumerical()},
        remainder="drop",
    )

    output = dispatch.fit_transform(_mixed_table())

    assert output.columns[Stype.numerical] == ("first",)
    assert torch.equal(output.numerical, torch.tensor([[1.0], [3.0]]))


def test_stype_dispatch_uses_route_fitted_state() -> None:
    dispatch = StypeDispatch({"numerical": StandardScale()}, remainder="drop")

    with pytest.raises(RuntimeError, match=r"StypeDispatch.*not fitted"):
        dispatch.transform(_mixed_table())

    transformed = dispatch.fit_transform(_mixed_table())
    assert torch.allclose(
        transformed.numerical.mean(dim=0),
        torch.zeros(2),
        atol=1e-6,
    )


def test_stype_dispatch_keeps_route_supported_stype_checks() -> None:
    dispatch = StypeDispatch(
        {"categorical": StandardScale()}, remainder="drop"
    )

    with pytest.raises(ValueError, match="categorical"):
        dispatch.fit_transform(_mixed_table())
