from collections.abc import Callable

import pytest
import torch
from sdm.processing import Clip, Power, Processor, Quantile, StandardScale
from sdm.processing.base import InvertibleMixin

ProcessorFactory = Callable[[], Processor]


def _quantile_processor() -> Quantile:
    return Quantile(n_quantiles=5, subsample=None)


@pytest.mark.parametrize(
    "processor_factory",
    [Clip, StandardScale],
)
def test_processor_requires_fit_for_transform(
    processor_factory: ProcessorFactory,
) -> None:
    processor = processor_factory()
    input = torch.ones(2, 2)

    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform(input)
    with pytest.raises(RuntimeError, match="not fitted"):
        processor(input)


@pytest.mark.parametrize(
    "processor_factory",
    [Clip, StandardScale],
)
def test_invertible_processor_requires_fit_for_inverse_transform(
    processor_factory: ProcessorFactory,
) -> None:
    processor = processor_factory()
    input = torch.ones(2, 2)

    assert isinstance(processor, InvertibleMixin)
    with pytest.raises(RuntimeError, match="not fitted"):
        processor.inverse_transform(input)


@pytest.mark.parametrize(
    "processor_factory",
    [
        Clip,
        StandardScale,
        Power,
        _quantile_processor,
    ],
)
def test_processor_transform_accepts_different_batch_size_after_fit(
    processor_factory: ProcessorFactory,
) -> None:
    fit_input = torch.tensor(
        [
            [-2.0, 0.0, 1.0],
            [-1.0, 10.0, 2.0],
            [0.0, 20.0, 3.0],
            [1.0, 30.0, 4.0],
            [2.0, 40.0, 5.0],
        ],
        dtype=torch.float64,
    )
    input = torch.tensor(
        [
            [-1.5, 5.0, 1.5],
            [0.5, 25.0, 4.5],
        ],
        dtype=torch.float64,
    )

    processor = processor_factory().fit(fit_input)
    transformed = processor.transform(input)

    assert transformed.shape == input.shape
    if isinstance(processor, InvertibleMixin):
        inverse = processor.inverse_transform(transformed)
        assert inverse.shape == input.shape


class StatelessProcessor(Processor):
    requires_fit = False

    def _transform(self, input: torch.Tensor) -> torch.Tensor:
        return input + 1


def test_stateless_processor_runs_without_fit() -> None:
    processor = StatelessProcessor()
    input = torch.ones(2, 2)

    assert torch.equal(processor.transform(input), input + 1)
    assert torch.equal(processor(input), input + 1)
    assert torch.equal(processor.fit_transform(input), input + 1)
