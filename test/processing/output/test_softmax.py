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


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_softmax_rejects_nonpositive_temperature(
    temperature: float,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        Softmax(temperature=temperature)


@withCUDA
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.float64])
def test_softmax_preserves_noncontiguous_logits(
    device: torch.device,
    temperature: float,
    dtype: torch.dtype,
) -> None:
    logits = torch.arange(12, dtype=dtype, device=device).view(3, 4).t()
    before = logits.clone()
    expected = (logits / temperature).softmax(dim=-1)

    output = Softmax(temperature=temperature).transform(
        TableTensor(numerical=logits)
    )

    torch.testing.assert_close(output.numerical, expected)
    torch.testing.assert_close(logits, before)


@withCUDA
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_softmax_preserves_autocast_dtype(
    device: torch.device,
    temperature: float,
    dtype: torch.dtype,
) -> None:
    logits = torch.tensor(
        [[-3.0, 0.0, 2.0], [4.0, -1.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    with torch.autocast(device.type, dtype=torch.bfloat16):
        expected = (logits / temperature).softmax(dim=-1)
        actual = Softmax(temperature=temperature).transform(
            TableTensor(numerical=logits)
        )

    torch.testing.assert_close(actual.numerical, expected)
