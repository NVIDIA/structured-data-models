import pytest
import torch
from sdm import TableTensor
from sdm.processing import MeanImpute
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
def test_mean_impute(device: torch.device, dtype: torch.dtype | None) -> None:
    inp = torch.tensor(
        [
            [1.0, torch.nan, torch.nan],
            [3.0, 5.0, torch.nan],
            [torch.nan, 7.0, torch.nan],
        ],
        dtype=dtype,
        device=device,
    )

    fill_value = -5.0
    processor = MeanImpute(fill_value=fill_value).fit(
        TableTensor.from_tensor(inp)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.allclose(
        processor._mean,
        torch.tensor([2.0, 6.0, fill_value], dtype=dtype, device=device),
    )
    assert torch.equal(
        transformed,
        torch.tensor(
            [
                [1.0, 6.0, fill_value],
                [3.0, 5.0, fill_value],
                [2.0, 7.0, fill_value],
            ],
            dtype=dtype,
            device=device,
        ),
    )


@pytest.mark.parametrize("fill_value", [torch.nan, torch.inf, -torch.inf])
def test_mean_impute_rejects_non_finite_fill_value(fill_value: float) -> None:
    with pytest.raises(ValueError, match="fill_value must be finite"):
        MeanImpute(fill_value=fill_value)
