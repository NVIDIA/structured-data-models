import pytest
import torch

from sdm import Stype, TableTensor
from sdm.processing import EnsembleInvertibleMixin, EnsembleProcessor
from sdm.tensor import EnsembleTable


# TODO: Replace these stubs with real EnsembleProcessor subclasses once they
# land, and exercise the EnsembleProcessor contract through those instead.
# Generator forwarding is only observable through a stochastic transformation
# and is therefore left to those processors as well.
class IdentityEnsembleProcessor(EnsembleProcessor):
    supported_stypes = frozenset({Stype.numerical})

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return ensemble_table


class InvertibleIdentityEnsembleProcessor(
    IdentityEnsembleProcessor,
    EnsembleInvertibleMixin,
):
    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return EnsembleTable(ensemble_table.table(0), num_members=1)


def test_ensemble_processor_preserves_member_order_and_metadata() -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[1.0], [2.0]]),
        columns=("first",),
    )
    second = TableTensor.from_tensor(
        torch.tensor([[3.0], [4.0]]),
        columns=("second",),
    )
    ensemble_table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(1, 0, 1),
    )
    processor = IdentityEnsembleProcessor()

    output = processor.fit_transform_ensemble(ensemble_table)

    assert output is ensemble_table
    assert output.num_members == 3
    assert output.table(0).columns == second.columns
    assert output.table(1).columns == first.columns
    assert output.table(2).columns == second.columns
    assert processor.transform_ensemble(ensemble_table) is ensemble_table


def test_ensemble_processor_requires_fit_before_transform() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    ensemble_table = EnsembleTable(table, num_members=2)
    processor = IdentityEnsembleProcessor()

    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform_ensemble(ensemble_table)

    assert processor.fit_ensemble(ensemble_table) is processor
    assert processor.transform_ensemble(ensemble_table) is ensemble_table


def test_ensemble_invertible_mixin_requires_fit_and_delegates() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    ensemble_table = EnsembleTable(table, num_members=2)
    processor = InvertibleIdentityEnsembleProcessor()

    with pytest.raises(RuntimeError, match="not fitted"):
        processor.inverse_transform_ensemble(ensemble_table)

    processor.fit_transform_ensemble(ensemble_table)
    output = processor.inverse_transform_ensemble(ensemble_table)

    assert output.num_members == 1
    assert output.table(0).equal(table)


def test_ensemble_processor_rejects_unsupported_stype() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1, dtype=torch.int64))
    ensemble_table = EnsembleTable(table, num_members=2)

    with pytest.raises(ValueError, match="categorical"):
        IdentityEnsembleProcessor().fit_transform_ensemble(ensemble_table)


def test_ensemble_processor_supports_table_tensor_lifecycle() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    fitted = IdentityEnsembleProcessor()
    processor = IdentityEnsembleProcessor()

    assert fitted.fit(table) is fitted
    assert fitted.transform(table).equal(table)
    assert processor.fit_transform(table).equal(table)
    assert processor.transform(table).equal(table)
    assert processor(table).equal(table)


def test_ensemble_processor_passthrough_for_empty_supported_blocks() -> None:
    empty_ensemble_table = EnsembleTable(
        TableTensor.from_tensor(torch.empty(2, 0)),
        num_members=2,
    )
    ensemble_table = EnsembleTable(
        TableTensor.from_tensor(torch.ones(2, 1)),
        num_members=2,
    )
    processor = IdentityEnsembleProcessor()

    assert processor.fit_ensemble(empty_ensemble_table) is processor
    assert (
        processor.fit_transform_ensemble(empty_ensemble_table)
        is empty_ensemble_table
    )
    assert (
        processor.transform_ensemble(empty_ensemble_table)
        is empty_ensemble_table
    )
    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform_ensemble(ensemble_table)
