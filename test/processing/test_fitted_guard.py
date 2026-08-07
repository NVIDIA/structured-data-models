from collections.abc import Callable

import pytest
import torch

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import PCA, ClipQuantiles, Processor, Standardize
from sdm.processing.base import InvertibleMixin

ProcessorFactory = Callable[[], Processor]


@pytest.mark.parametrize(
    "processor_factory",
    [ClipQuantiles, Standardize],
)
def test_processor_requires_fit_for_transform(
    processor_factory: ProcessorFactory,
) -> None:
    processor = processor_factory()
    inp = TableTensor.from_tensor(torch.ones(2, 2))

    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform(inp)
    with pytest.raises(RuntimeError, match="not fitted"):
        processor(inp)


@pytest.mark.parametrize(
    "processor_factory",
    [Standardize],
)
def test_invertible_processor_requires_fit_for_inverse_transform(
    processor_factory: ProcessorFactory,
) -> None:
    processor = processor_factory()
    inp = TableTensor.from_tensor(torch.ones(2, 2))

    assert isinstance(processor, InvertibleMixin)
    with pytest.raises(RuntimeError, match="not fitted"):
        processor.inverse_transform(inp)


class StatelessProcessor(Processor):
    operates_on_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical + 1)


def test_stateless_processor_runs_without_fit() -> None:
    processor = StatelessProcessor()
    inp = torch.ones(2, 2)
    table = TableTensor.from_tensor(inp)

    assert torch.equal(processor.transform(table).numerical, inp + 1)
    assert torch.equal(processor(table).numerical, inp + 1)
    assert torch.equal(processor.fit_transform(table).numerical, inp + 1)


def _mixed_table() -> TableTensor:
    return TableTensor(
        columns={
            "numerical": ("x0",),
            "categorical": ("kind",),
        },
        numerical=torch.tensor([[1.0], [2.0]]),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )


def _id_table() -> TableTensor:
    return TableTensor(
        columns={"id": ("row_id",)},
        id=ColumnarTensor((torch.arange(2),)),
    )


def test_processor_preserves_unoperated_stypes_on_forward_paths() -> None:
    mixed = _mixed_table()

    transformed = Standardize().fit_transform(mixed)
    processor = Standardize().fit(mixed)

    for output in (transformed, processor.transform(mixed), processor(mixed)):
        assert output.columns == mixed.columns
        torch.testing.assert_close(
            output.numerical.mean(dim=0),
            torch.zeros(1),
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.equal(output.categorical.code, mixed.categorical.code)


def test_processor_noops_when_only_unoperated_stypes_are_active() -> None:
    table = _id_table()

    processor = Standardize()

    assert processor.fit(table) is processor
    assert processor.fit_transform(table) is table
    assert processor.transform(table) is table


def test_processor_fit_transform_handles_empty_table() -> None:
    table = TableTensor.from_tensor(torch.empty(3, 0))
    output = PCA(num_components=2).fit_transform(table)

    assert output.size() == table.size()
    assert output.schema == table.schema
