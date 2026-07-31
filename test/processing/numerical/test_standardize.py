import torch

from sdm import TableTensor
from sdm.processing import Standardize
from sdm.testing import withCUDA


@withCUDA
def test_standardize_fit_transform_and_inverse_round_trip(
    device: torch.device,
) -> None:
    inp = TableTensor.from_tensor(
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

    processor = Standardize().fit(inp)
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
    expected = (inp.numerical - expected_mean) / expected_scale

    assert torch.allclose(processor.mean, expected_mean)
    assert torch.allclose(processor.scale, expected_scale)
    transformed = processor.transform(inp).numerical
    assert torch.allclose(transformed, expected)
    assert transformed.device == device
    assert torch.allclose(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp.numerical,
    )


@withCUDA
def test_standardize_without_mean_or_std(device: torch.device) -> None:
    inp = torch.tensor([[1.0, 2.0], [3.0, 6.0]], device=device)

    processor = Standardize(with_mean=False, with_std=False).fit(
        TableTensor.from_tensor(inp)
    )

    assert torch.equal(processor.mean, torch.zeros(2, device=device))
    assert torch.equal(processor.scale, torch.ones(2, device=device))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical
    assert torch.equal(transformed, inp)
    assert transformed.device == device


@withCUDA
def test_standardize_single_sample_uses_unit_scale(
    device: torch.device,
) -> None:
    inp = torch.tensor([[42.0, -2.0]], device=device)

    processor = Standardize().fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.equal(processor.scale, torch.ones(2, device=device))
    assert torch.equal(transformed, torch.zeros_like(inp))
    assert transformed.device == device
    assert torch.equal(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
    )


@withCUDA
def test_standardize_fits_leading_batches_independently(
    device: torch.device,
) -> None:
    context = torch.tensor(
        [[[1.0], [3.0]], [[10.0], [14.0]]],
        device=device,
    )
    query = torch.tensor([[[4.0]], [[16.0]]], device=device)

    processor = Standardize().fit(TableTensor.from_tensor(context))
    transformed = processor.transform(TableTensor.from_tensor(query)).numerical

    assert torch.equal(
        processor.mean,
        torch.tensor([[[2.0]], [[12.0]]], device=device),
    )
    assert torch.equal(
        processor.scale,
        torch.tensor([[[1.0]], [[2.0]]], device=device),
    )
    assert torch.equal(transformed, torch.full_like(query, 2.0))
    assert torch.equal(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        query,
    )
