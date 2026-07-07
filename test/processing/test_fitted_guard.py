from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
from sdm import TableTensor
from sdm.processing import Clip, Processor, StandardScale
from sdm.processing.base import InvertibleMixin

ProcessorFactory = Callable[[], Processor]


@pytest.mark.parametrize(
    "processor_factory",
    [Clip, StandardScale],
)
def test_processor_requires_fit_for_transform(
    processor_factory: ProcessorFactory,
) -> None:
    processor = processor_factory()
    input = TableTensor.from_tensor(torch.ones(2, 2))

    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform(input)


@pytest.mark.parametrize(
    "processor_factory",
    [Clip, StandardScale],
)
def test_invertible_processor_requires_fit_for_inverse_transform(
    processor_factory: ProcessorFactory,
) -> None:
    processor = processor_factory()
    input = TableTensor.from_tensor(torch.ones(2, 2))

    assert isinstance(processor, InvertibleMixin)
    with pytest.raises(RuntimeError, match="not fitted"):
        processor.inverse_transform(input)


class StatelessProcessor(Processor):
    requires_fit = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + 1


def test_stateless_processor_runs_without_fit() -> None:
    processor = StatelessProcessor()
    input = torch.ones(2, 2)
    table = TableTensor.from_tensor(input)

    assert torch.equal(processor.transform(table).numerical, input + 1)
    assert torch.equal(processor(table).numerical, input + 1)
    assert torch.equal(processor.fit_transform(table).numerical, input + 1)

