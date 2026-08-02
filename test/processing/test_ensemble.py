import pytest
import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import (
    EnsembleProcessor,
    EnsembleProcessorAdapter,
    InvertibleMixin,
    Processor,
)


class IdentityEnsembleProcessor(EnsembleProcessor):
    supported_stypes = frozenset({Stype.numerical})

    def __init__(self) -> None:
        super().__init__()
        self.generator: torch.Generator | None = None

    def _fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self.generator = generator
        return table

    def _transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        return table


class ExpandingEnsembleProcessor(IdentityEnsembleProcessor):
    requires_fit = False

    def _transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        return EnsembleTable(table.representation(0), num_members=2)


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


def test_ensemble_processor_is_a_processor() -> None:
    assert issubclass(EnsembleProcessor, Processor)


def test_ensemble_processor_preserves_member_order_and_metadata() -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[1.0], [2.0]]),
        columns=("first",),
    )
    second = TableTensor.from_tensor(
        torch.tensor([[3.0], [4.0]]),
        columns=("second",),
    )
    table = EnsembleTable.from_representations(
        representations=(first, second),
        member_representation_ids=(1, 0, 1),
    )
    generator = torch.Generator()
    processor = IdentityEnsembleProcessor()

    output = processor.fit_transform_ensemble(
        table,
        generator=generator,
    )

    assert processor.generator is generator
    assert output is table
    assert output.num_members == 3
    assert output.representation(0).columns == second.columns
    assert output.representation(1).columns == first.columns
    assert output.representation(2).columns == second.columns
    assert processor.transform_ensemble(table) is table


def test_ensemble_processor_requires_fit_before_transform() -> None:
    data = TableTensor.from_tensor(torch.ones(2, 1))
    table = EnsembleTable(data, num_members=2)
    processor = IdentityEnsembleProcessor()

    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform_ensemble(table)


def test_ensemble_processor_rejects_unsupported_stype() -> None:
    data = TableTensor.from_tensor(torch.ones(2, 1, dtype=torch.int64))
    table = EnsembleTable(data, num_members=2)

    with pytest.raises(ValueError, match="categorical"):
        IdentityEnsembleProcessor().fit_transform_ensemble(table)


def test_ensemble_processor_supports_table_tensor_lifecycle() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    fitted = IdentityEnsembleProcessor()
    processor = IdentityEnsembleProcessor()

    assert fitted.fit(table) is fitted
    assert fitted.transform(table).equal(table)
    assert processor.fit_transform(table).equal(table)
    assert processor.transform(table).equal(table)
    assert processor(table).equal(table)


def test_only_ensemble_api_accepts_multiple_output_members() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    ensemble = EnsembleTable(table, num_members=1)
    processor = ExpandingEnsembleProcessor()

    assert processor.transform_ensemble(ensemble).num_members == 2
    with pytest.raises(RuntimeError, match="exactly one member"):
        processor.transform(table)


def test_adapter_preserves_packed_member_mapping() -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[1.0], [3.0]]), columns=("first",)
    )
    second = TableTensor.from_tensor(
        torch.tensor([[2.0], [6.0]]), columns=("second",)
    )
    table = EnsembleTable.from_representations(
        (first, second),
        member_representation_ids=(1, 0, 1),
    )
    processor = EnsembleProcessorAdapter(Center())

    output = processor.fit_transform_ensemble(table)

    assert output.num_members == 3
    assert output.representation(0).numerical.tolist() == [[-2.0], [2.0]]
    assert output.representation(1).numerical.tolist() == [[-1.0], [1.0]]
    assert output.representation(2).equal(output.representation(0))
    restored = processor.inverse_transform_ensemble(output)
    for member_id in range(table.num_members):
        assert restored.representation(member_id).equal(
            table.representation(member_id)
        )


def test_adapter_returns_ensemble_processors_unchanged() -> None:
    processor = IdentityEnsembleProcessor()
    assert EnsembleProcessorAdapter.adapt(processor) is processor


def test_adapter_rejects_row_changing_processor() -> None:
    table = EnsembleTable(
        TableTensor.from_tensor(torch.ones(2, 1)),
        num_members=2,
    )
    processor = EnsembleProcessorAdapter(
        Processor.as_processor(lambda value: value[..., :1, :])
    )

    with pytest.raises(ValueError, match="row dimension"):
        processor.fit_transform_ensemble(table)
