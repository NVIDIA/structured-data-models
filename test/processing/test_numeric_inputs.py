import pytest
import torch
from sdm import TableTensor
from sdm.processing import (
    Clip,
    MeanImpute,
    Power,
    Quantile,
    SigmaClip,
    SoftmaxTemperature,
    StandardScale,
)


@pytest.mark.parametrize(
    "processor",
    [
        Clip(),
        MeanImpute(),
        Power(),
        Quantile(n_quantiles=4, subsample=None),
        SigmaClip(),
        SoftmaxTemperature(),
        StandardScale(),
    ],
)
def test_numeric_processors_accept_integer_input(processor) -> None:
    inp = torch.tensor(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ]
    )

    output = processor.fit_transform(TableTensor.from_tensor(inp)).numerical

    assert output.dtype == torch.get_default_dtype()
    assert output.shape == inp.shape


def test_numeric_processors_promote_integer_input_to_default_dtype() -> None:
    inp = torch.tensor([[1, 2], [3, 4]])
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        output = (
            StandardScale()
            .fit_transform(TableTensor.from_tensor(inp))
            .numerical
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    assert output.dtype == torch.float64
