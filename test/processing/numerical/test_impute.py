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
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_impute_mean_matches_nanmean_with_large_counts(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    numerical = torch.ones((70_001, 4), dtype=dtype, device=device)
    numerical[::17, 0] = torch.nan
    numerical[:, 1] = torch.nan
    numerical[0, 2] = torch.inf
    numerical[0, 3] = -torch.inf
    expected = numerical.nanmean(dim=-2, keepdim=True)
    expected.masked_fill_(expected.isnan(), -5.0)

    processor = ImputeMean(fill_value=-5.0).fit(
        TableTensor.from_tensor(numerical)
    )
    query = torch.full_like(expected, torch.nan)
    actual = processor.transform(TableTensor.from_tensor(query)).numerical

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert query.isnan().all()
