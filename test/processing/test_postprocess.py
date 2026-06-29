import pytest
import torch
from sdm.processing import SoftmaxTemperature
from sdm.testing import withCUDA


@withCUDA
def test_softmax_temperature_controls_sharpness(
    device: torch.device,
) -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]], device=device)

    colder = SoftmaxTemperature(temperature=0.5).transform(logits)
    warmer = SoftmaxTemperature(temperature=2.0).transform(logits)

    assert colder[0, -1] > warmer[0, -1]
    assert colder[0, 0] < warmer[0, 0]


@withCUDA
def test_softmax_temperature_is_numerically_stable(
    device: torch.device,
) -> None:
    logits = torch.tensor(
        [[1000.0, 1001.0], [-1000.0, -1001.0]],
        device=device,
    )

    output = SoftmaxTemperature().transform(logits)

    assert torch.isfinite(output).all()
    assert torch.allclose(output.sum(dim=-1), torch.ones(2, device=device))


@pytest.mark.parametrize("temperature", [0.0, float("inf"), float("nan")])
def test_softmax_temperature_rejects_invalid_temperature(
    temperature: float,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        SoftmaxTemperature(temperature=temperature)
