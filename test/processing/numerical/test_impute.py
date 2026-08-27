import pytest
import torch

from sdm import TableTensor
from sdm.processing import ImputeMean
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
def test_impute_mean(device: torch.device, dtype: torch.dtype | None) -> None:
    inp = torch.tensor(
        [
            [
                [1.0, torch.nan, torch.nan],
                [3.0, 5.0, torch.nan],
                [torch.nan, 7.0, torch.nan],
            ],
            [
                [10.0, 2.0, torch.nan],
                [20.0, torch.nan, torch.nan],
                [30.0, 6.0, torch.nan],
            ],
        ],
        dtype=dtype,
        device=device,
    )

    fill_value = -5.0
    processor = ImputeMean(fill_value=fill_value).fit(
        TableTensor.from_tensor(inp)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.equal(
        transformed,
        torch.tensor(
            [
                [
                    [1.0, 6.0, fill_value],
                    [3.0, 5.0, fill_value],
                    [2.0, 7.0, fill_value],
                ],
                [
                    [10.0, 2.0, fill_value],
                    [20.0, 4.0, fill_value],
                    [30.0, 6.0, fill_value],
                ],
            ],
            dtype=dtype,
            device=device,
        ),
    )


@withCUDA
def test_impute_mean_can_treat_infinity_as_missing(
    device: torch.device,
) -> None:
    context = TableTensor.from_tensor(
        torch.tensor(
            [[1.0, torch.inf], [3.0, -torch.inf], [torch.nan, torch.nan]],
            device=device,
        )
    )
    query = TableTensor.from_tensor(
        torch.tensor(
            [[torch.inf, -torch.inf], [5.0, torch.nan]], device=device
        )
    )

    out = ImputeMean(nonfinite=True).fit(context).transform(query)

    assert torch.equal(
        out.numerical,
        torch.tensor([[2.0, 0.0], [5.0, 0.0]], device=device),
    )
    assert repr(ImputeMean()) == "ImputeMean()"
    assert repr(ImputeMean(fill_value=-1.0, nonfinite=True)) == (
        "ImputeMean(fill_value=-1.0, nonfinite=True)"
    )
