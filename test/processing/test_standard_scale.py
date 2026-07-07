import torch
from sdm import TableTensor
from sdm.processing import StandardScale
from sdm.testing import withCUDA


@withCUDA
def test_standard_scale_fit_transform_and_inverse_round_trip(
    device: torch.device,
) -> None:
    input = TableTensor.from_tensor(
        torch.tensor(
            [
                [1.0, 2.0, 5.0],
                [3.0, 2.0, 7.0],
                [5.0, 2.0, 9.0],
            ],
            dtype=torch.float64,
            device=device,
        )
    )

    processor = StandardScale().fit(input)
    expected_mean = torch.tensor(
        [3.0, 2.0, 7.0],
        dtype=torch.float64,
        device=device,
    )
    expected_scale = torch.tensor(
        [
            torch.sqrt(torch.tensor(8.0 / 3.0)),
            1.0,
            torch.sqrt(torch.tensor(8.0 / 3.0)),
        ],
        dtype=torch.float64,
        device=device,
    )
    expected = (input.numerical - expected_mean) / expected_scale

    assert torch.allclose(processor.mean, expected_mean)
    assert torch.allclose(processor.scale, expected_scale)
    transformed = processor.transform(input).numerical
    assert torch.allclose(transformed, expected)
    assert transformed.device == device
    assert torch.allclose(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        input.numerical,
    )


@withCUDA
def test_standard_scale_without_mean_or_std(device: torch.device) -> None:
    input = torch.tensor([[1.0, 2.0], [3.0, 6.0]], device=device)

    processor = StandardScale(with_mean=False, with_std=False).fit(
        TableTensor.from_tensor(input)
    )

    assert torch.equal(processor.mean, torch.zeros(2, device=device))
    assert torch.equal(processor.scale, torch.ones(2, device=device))
    transformed = processor.transform(TableTensor.from_tensor(input)).numerical
    assert torch.equal(transformed, input)
    assert transformed.device == device


@withCUDA
def test_standard_scale_nan_columns_follow_torch_reductions(
    device: torch.device,
) -> None:
    input = torch.tensor(
        [
            [1.0, 1.0],
            [torch.nan, 3.0],
            [5.0, 5.0],
        ],
        device=device,
    )

    processor = StandardScale().fit(TableTensor.from_tensor(input))
    transformed = processor.transform(TableTensor.from_tensor(input)).numerical

    assert torch.isnan(processor.mean[0])
    assert torch.isnan(processor.scale[0])
    assert torch.isnan(transformed[:, 0]).all()
    assert torch.isfinite(transformed[:, 1]).all()
    assert transformed.device == device


@withCUDA
def test_standard_scale_single_sample_uses_unit_scale(
    device: torch.device,
) -> None:
    input = torch.tensor([[42.0, -2.0]], device=device))

    processor = StandardScale().fit(TableTensor.from_tensor(input))
    transformed = processor.transform(TableTensor.from_tensor(input)).numerical

    assert torch.equal(processor.scale, torch.ones(2, device=device))
    assert torch.equal(transformed, torch.zeros_like(input))
    assert transformed.device == device
    assert torch.equal(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        input,
    )
