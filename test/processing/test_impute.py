import torch
from sdm.processing import MeanImpute
from sdm.testing import withCUDA


@withCUDA
def test_mean_impute_replaces_nan_with_column_mean(
    device: torch.device,
) -> None:
    input = torch.tensor(
        [
            [1.0, torch.nan],
            [3.0, 5.0],
            [torch.nan, 7.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    processor = MeanImpute().fit(input)
    transformed = processor.transform(input)

    assert torch.allclose(
        processor.mean,
        torch.tensor([2.0, 6.0], dtype=torch.float64, device=device),
    )
    assert torch.equal(
        transformed,
        torch.tensor(
            [
                [1.0, 6.0],
                [3.0, 5.0],
                [2.0, 7.0],
            ],
            dtype=torch.float64,
            device=device,
        ),
    )
    assert transformed.device == device


@withCUDA
def test_mean_impute_no_nan_keeps_values(device: torch.device) -> None:
    input = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float64,
        device=device,
    )

    processor = MeanImpute().fit(input)
    transformed = processor.transform(input)

    assert torch.equal(transformed, input)
    assert transformed.device == device


@withCUDA
def test_mean_impute_all_nan_column_uses_fill_value_and_keeps_shape(
    device: torch.device,
) -> None:
    input = torch.tensor(
        [
            [torch.nan, 1.0],
            [torch.nan, 3.0],
        ],
        device=device,
    )

    processor = MeanImpute(fill_value=-5.0).fit(input)
    transformed = processor.transform(input)

    assert transformed.shape == input.shape
    assert torch.equal(
        processor.mean, torch.tensor([-5.0, 2.0], device=device)
    )
    assert torch.equal(
        transformed,
        torch.tensor(
            [
                [-5.0, 1.0],
                [-5.0, 3.0],
            ],
            device=device,
        ),
    )
    assert transformed.device == device


@withCUDA
def test_mean_impute_preserves_floating_dtype(device: torch.device) -> None:
    input = torch.tensor(
        [[1.0], [torch.nan]], dtype=torch.float32, device=device
    )

    transformed = MeanImpute().fit(input).transform(input)

    assert transformed.dtype == torch.float32
    assert torch.equal(
        transformed,
        torch.tensor([[1.0], [1.0]], dtype=torch.float32, device=device),
    )
    assert transformed.device == device
