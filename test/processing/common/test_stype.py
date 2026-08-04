import pytest
import torch

from sdm import (
    CategoricalTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import (
    EnsembleProcessor,
    Identity,
    ImputeMean,
    InvertibleMixin,
    Processor,
    ShuffleCategories,
    Standardize,
    StypeDispatch,
)
from sdm.tensor import EnsembleTable


class Center(Processor, InvertibleMixin):
    supported_stypes = frozenset({Stype.numerical})

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.empty(0))

    def _fit(self, table: TableTensor, **_: object) -> None:
        self.mean = table.numerical.mean(dim=-2, keepdim=True)

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical - self.mean)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical + self.mean)


class ExpandMembers(EnsembleProcessor):
    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        return self._transform_ensemble(ensemble_table)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return EnsembleTable(
            ensemble_table.table(0),
            num_members=ensemble_table.num_members + 1,
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


def test_stype_dispatch_inverse_rejects_dropped_remainder() -> None:
    table = _mixed_table()
    dispatch = StypeDispatch(
        numerical=Standardize(),
        remainder="drop",
    )

    output = dispatch.fit_transform(table)

    with pytest.raises(ValueError, match="remainder='drop'"):
        dispatch.inverse_transform(output)


def test_stype_dispatch_rejects_remainder_before_fitting_routes() -> None:
    table = _mixed_table()
    processor = Standardize()
    dispatch = StypeDispatch(
        numerical=processor,
        remainder="error",
    )

    with pytest.raises(ValueError, match=r"non-empty.*categorical.*no route"):
        dispatch.fit(table)

    with pytest.raises(RuntimeError, match=r"Standardize.*not fitted"):
        processor.transform(table.select_stypes(Stype.numerical))


def test_stype_dispatch_drops_remainder_and_empty_outputs() -> None:
    output = StypeDispatch(remainder="drop").fit_transform(_mixed_table())

    assert output.size() == (2, 0)
    assert output.columns == {
        Stype.numerical: (),
        Stype.categorical: (),
        Stype.datetime: (),
        Stype.text: (),
        Stype.id: (),
    }


def test_stype_dispatch_runs_iterable_routes() -> None:
    table = _mixed_table().replace_blocks(
        numerical=torch.tensor([[-3.0, 2.0], [1.0, 4.0]])
    )
    dispatch = StypeDispatch(
        numerical=[
            lambda table: table.replace_blocks(
                numerical=table.numerical.square()
            ),
            ImputeMean(),
            Standardize(),
        ],
        remainder="drop",
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
    dispatch = StypeDispatch(
        numerical=Standardize(),
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


def test_stype_dispatch_ensemble_routes_members_and_preserves_order() -> None:
    first = _mixed_table()
    second = _mixed_table().replace_blocks(
        numerical=torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    )
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(1, 0, 1),
    )
    processor = StypeDispatch(
        numerical=lambda value: value.replace_blocks(
            numerical=value.numerical.square()
        )
    )

    output = processor.transform_ensemble(table)

    for member_id, source in enumerate((second, first, second)):
        result = output.table(member_id)
        assert torch.equal(result.numerical, source.numerical.square())
        assert torch.equal(result.categorical.code, source.categorical.code)


def test_stype_dispatch_ensemble_keeps_fitted_state_per_group() -> None:
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
    processor = StypeDispatch(numerical=Center())

    transformed = processor.fit_transform_ensemble(table)
    query = processor.transform_ensemble(table)
    restored = processor.inverse_transform_ensemble(transformed)

    for member_id in range(table.num_members):
        result = transformed.table(member_id)
        assert torch.allclose(
            result.numerical.mean(dim=-2),
            torch.zeros(1),
        )
        assert query.table(member_id).equal(result)
        assert restored.table(member_id).columns == (
            table.table(member_id).columns
        )
        assert restored.table(member_id).equal(table.table(member_id))


def test_stype_dispatch_fit_ensemble_fits_routes() -> None:
    table = EnsembleTable(_mixed_table(), num_members=2)
    fitted = StypeDispatch(numerical=Center())
    combined = StypeDispatch(numerical=Center())

    fitted.fit_ensemble(table)
    transformed = fitted.transform_ensemble(table)
    expected = combined.fit_transform_ensemble(table)

    for member_id in range(table.num_members):
        assert transformed.table(member_id).equal(expected.table(member_id))


def test_stype_dispatch_ensemble_inverse_rejects_drop() -> None:
    table = EnsembleTable(_mixed_table(), num_members=2)
    processor = StypeDispatch(numerical=Identity(), remainder="drop")
    transformed = processor.fit_transform_ensemble(table)

    with pytest.raises(ValueError, match="not invertible"):
        processor.inverse_transform_ensemble(transformed)


def test_stype_dispatch_stateless_ensemble_inverse() -> None:
    table = EnsembleTable(_mixed_table(), num_members=2)
    processor = StypeDispatch(numerical=Identity())

    transformed = processor.transform_ensemble(table)
    restored = processor.inverse_transform_ensemble(transformed)

    for member_id in range(table.num_members):
        assert restored.table(member_id).equal(table.table(member_id))


def test_stype_dispatch_rejects_route_member_count_change() -> None:
    table = EnsembleTable(_mixed_table(), num_members=2)
    processor = StypeDispatch(numerical=ExpandMembers())

    with pytest.raises(ValueError, match="preserve ensemble member count"):
        processor.transform_ensemble(table)


def test_stype_dispatch_ensemble_inverse_rejects_non_invertible_route() -> (
    None
):
    table = EnsembleTable(_mixed_table(), num_members=2)
    processor = StypeDispatch(numerical=ImputeMean())
    transformed = processor.fit_transform_ensemble(table)

    with pytest.raises(TypeError, match="not invertible"):
        processor.inverse_transform_ensemble(transformed)
