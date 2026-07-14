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
from sdm.processing import Processor, QuantileClip, StandardScale
from sdm.processing.base import InvertibleMixin

ProcessorFactory = Callable[[], Processor]


@pytest.mark.parametrize(
    "processor_factory",
    [QuantileClip, StandardScale],
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
    [QuantileClip, StandardScale],
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
    supported_stypes = frozenset({Stype.numerical})
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
            data=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )


def _id_table() -> TableTensor:
    return TableTensor(
        columns={"id": ("row_id",)},
        id=ColumnarTensor((torch.arange(2),)),
    )


def test_processor_rejects_unsupported_stype_on_forward_paths() -> None:
    numerical = TableTensor.from_tensor(torch.ones(2, 1))
    mixed = _mixed_table()
    processor = StandardScale().fit(numerical)

    with pytest.raises(ValueError, match="categorical"):
        StandardScale().fit(mixed)
    with pytest.raises(ValueError, match="categorical"):
        processor.transform(mixed)
    with pytest.raises(ValueError, match="categorical"):
        processor(mixed)


def test_processor_rejects_id_stype() -> None:
    with pytest.raises(ValueError, match="id"):
        StandardScale().fit(_id_table())
