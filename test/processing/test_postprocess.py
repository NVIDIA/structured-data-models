import pytest
import torch
from sdm.processing import SoftmaxTemperature


def test_softmax_temperature_runs_without_fit() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]])

    processor = SoftmaxTemperature()

    assert torch.allclose(
        processor.transform(logits), torch.softmax(logits, dim=-1)
    )


def test_softmax_temperature_controls_sharpness() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]])

    colder = SoftmaxTemperature(temperature=0.5).fit(logits).transform(logits)
    warmer = SoftmaxTemperature(temperature=2.0).fit(logits).transform(logits)

    assert colder[0, -1] > warmer[0, -1]
    assert colder[0, 0] < warmer[0, 0]


def test_softmax_temperature_is_numerically_stable() -> None:
    logits = torch.tensor([[1000.0, 1001.0], [-1000.0, -1001.0]])

    output = SoftmaxTemperature().fit(logits).transform(logits)

    assert torch.isfinite(output).all()
    assert torch.allclose(output.sum(dim=-1), torch.ones(2))


def test_softmax_temperature_matches_tabicl_numpy_formula() -> None:
    logits = torch.tensor(
        [
            [1.5, -2.0, 0.25],
            [100.0, 101.0, 99.0],
        ],
        dtype=torch.float64,
    )
    temperature = 0.9
    scaled = logits / temperature
    exp = torch.exp(scaled - scaled.max(dim=-1, keepdim=True).values)
    expected = exp / exp.sum(dim=-1, keepdim=True)

    output = (
        SoftmaxTemperature(temperature=temperature)
        .fit(logits)
        .transform(logits)
    )

    assert torch.allclose(output, expected)


def test_softmax_temperature_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="positive"):
        SoftmaxTemperature(temperature=0.0)


