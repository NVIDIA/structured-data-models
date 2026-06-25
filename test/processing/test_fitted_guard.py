from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from schemafm.processing import Clip, Power, Processor, Quantile, StandardScale
from schemafm.processing.base import InvertibleMixin

ProcessorFactory = Callable[[], Processor]


@pytest.mark.parametrize(
    "processor_factory",
    [Clip, Power, Quantile, StandardScale],
)
def test_processor_requires_fit_for_transform(
    processor_factory: ProcessorFactory,
) -> None:
    processor = processor_factory()
    input = torch.ones(2, 2)

    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform(input)


@pytest.mark.parametrize(
    "processor_factory",
    [Clip, Power, Quantile, StandardScale],
)
def test_invertible_processor_requires_fit_for_inverse_transform(
    processor_factory: ProcessorFactory,
) -> None:
    processor = processor_factory()
    input = torch.ones(2, 2)

    assert isinstance(processor, InvertibleMixin)
    with pytest.raises(RuntimeError, match="not fitted"):
        processor.inverse_transform(input)


class StatelessProcessor(Processor):
    requires_fit = False

    def forward(
        self, input: torch.Tensor, offset: torch.Tensor | int = 1
    ) -> torch.Tensor:
        return input + offset


def test_stateless_processor_runs_without_fit() -> None:
    processor = StatelessProcessor()
    input = torch.ones(2, 2)

    assert torch.equal(processor.transform(input), input + 1)
    assert torch.equal(processor(input), input + 1)

    offset = input.new_full((), 2)
    assert torch.equal(processor.transform(input, offset), input + offset)
    assert torch.equal(processor(input, offset), input + offset)
