import torch
from typing_extensions import Self

from sdm import Stype, TableTensor
from sdm.processing import Processor, VariableSchemaBatchMixin


class _VariableSchemaProcessor(Processor, VariableSchemaBatchMixin):
    supported_stypes = frozenset({Stype.numerical})

    def __init__(self) -> None:
        super().__init__()
        self.fitted_table: TableTensor | None = None
        self.generator: torch.Generator | None = None

    def _transform(self, table: TableTensor) -> TableTensor:
        return table

    def fit_batch(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Self:
        self.fitted_table = table
        self.generator = generator
        return self

    def transform_batch(
        self,
        table: TableTensor,
    ) -> tuple[TableTensor, ...]:
        return tuple(table[index] for index in range(table.size(0)))


def test_variable_schema_batch_mixin_fits_and_transforms_each_entry() -> None:
    table = TableTensor.from_tensor(torch.arange(12).reshape(2, 3, 2))
    generator = torch.Generator().manual_seed(0)
    processor = _VariableSchemaProcessor()

    output = processor.fit_transform_batch(table, generator=generator)

    assert processor.fitted_table is table
    assert processor.generator is generator
    assert len(output) == table.size(0)
    assert all(
        torch.equal(out.numerical, table[index].numerical)
        for index, out in enumerate(output)
    )
