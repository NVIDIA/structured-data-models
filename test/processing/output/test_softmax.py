import pytest
import torch

from sdm import TableTensor
from sdm.processing import Softmax
from sdm.testing import withCUDA


@withCUDA
def test_softmax_controls_sharpness(
    device: torch.device,
) -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]], device=device)

    colder = (
        Softmax(temperature=0.5)
        .transform(TableTensor.from_tensor(logits))
        .numerical
    )
    warmer = (
        Softmax(temperature=2.0)
        .transform(TableTensor.from_tensor(logits))
        .numerical
    )

    assert colder[0, -1] > warmer[0, -1]
    assert colder[0, 0] < warmer[0, 0]


@withCUDA
def test_softmax_is_numerically_stable(
    device: torch.device,
) -> None:
    logits = torch.tensor(
        [[1000.0, 1001.0], [-1000.0, -1001.0]],
        device=device,
    )

    output = Softmax().transform(TableTensor.from_tensor(logits)).numerical

    assert torch.isfinite(output).all()
    assert torch.allclose(output.sum(dim=-1), torch.ones(2, device=device))


@pytest.mark.parametrize("temperature", [0.0, float("inf"), float("nan")])
def test_softmax_rejects_invalid_temperature(
    temperature: float,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        Softmax(temperature=temperature)
