import pytest
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
        [[3.0, 2.0, 7.0]],
        dtype=torch.float64,
        device=device,
    )
    expected_scale = torch.tensor(
        [
            [
                torch.sqrt(torch.tensor(8.0 / 3.0)),
                1.0,
                torch.sqrt(torch.tensor(8.0 / 3.0)),
            ]
        ],
        dtype=torch.float64,
        device=device,
    )
    expected = (inp.numerical - expected_mean) / expected_scale

    assert torch.allclose(processor.mean, expected_mean)
    assert torch.allclose(processor.scale, expected_scale)
    out = processor.transform(inp)
    assert torch.allclose(out.numerical, expected)
    assert torch.allclose(
        processor.inverse_transform(out).numerical,
        inp.numerical,
    )


@withCUDA
@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_standardize_with_mean_and_std_options(
    device: torch.device,
    with_mean: bool,
    with_std: bool,
) -> None:
    inp = torch.tensor(
        [
            [1.0, 2.0, float("nan")],
            [3.0, 6.0, float("inf")],
            [float("nan"), float("inf"), -float("inf")],
        ],
        device=device,
    )

    processor = Standardize(with_mean=with_mean, with_std=with_std)
    processor.fit(TableTensor.from_tensor(inp))

    if with_mean:
        expected_mean = torch.tensor([[2.0, 4.0, 0.0]], device=device)
    else:
        expected_mean = torch.tensor([[0.0, 0.0, 0.0]], device=device)

    if with_std:
        expected_scale = torch.tensor([[1.0, 2.0, 1.0]], device=device)
    else:
        expected_scale = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    torch.testing.assert_close(processor.mean, expected_mean)
    torch.testing.assert_close(processor.scale, expected_scale)
    out = processor.transform(TableTensor.from_tensor(inp))

    if with_mean and with_std:
        expected = [
            [-1.0, -1.0, float("nan")],
            [1.0, 1.0, float("inf")],
            [float("nan"), float("inf"), -float("inf")],
        ]
    elif with_mean:
        expected = [
            [-1.0, -2.0, float("nan")],
            [1.0, 2.0, float("inf")],
            [float("nan"), float("inf"), -float("inf")],
        ]
    elif with_std:
        expected = [
            [1.0, 1.0, float("nan")],
            [3.0, 3.0, float("inf")],
            [float("nan"), float("inf"), -float("inf")],
        ]
    else:
        expected = [
            [1.0, 2.0, float("nan")],
            [3.0, 6.0, float("inf")],
            [float("nan"), float("inf"), -float("inf")],
        ]

    torch.testing.assert_close(
        out.numerical,
        torch.tensor(expected, device=device),
        equal_nan=True,
    )
    torch.testing.assert_close(
        processor.inverse_transform(out).numerical,
        inp,
        equal_nan=True,
    )


@withCUDA
def test_standardize_single_sample_uses_unit_scale(
    device: torch.device,
) -> None:
    inp = torch.tensor([[42.0, -2.0]], device=device)

    processor = Standardize().fit(TableTensor.from_tensor(inp))
    out = processor.transform(TableTensor.from_tensor(inp))

    assert torch.equal(processor.scale, torch.ones((1, 2), device=device))
    assert torch.equal(out.numerical, torch.zeros_like(inp))
    assert torch.equal(
        processor.inverse_transform(out).numerical,
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
    out = processor.transform(TableTensor.from_tensor(query))

    assert torch.equal(
        processor.mean,
        torch.tensor([[[2.0]], [[12.0]]], device=device),
    )
    assert torch.equal(
        processor.scale,
        torch.tensor([[[1.0]], [[2.0]]], device=device),
    )
    assert torch.equal(out.numerical, torch.full_like(query, 2.0))
    assert torch.equal(
        processor.inverse_transform(out).numerical,
        query,
    )


@withCUDA
def test_standardize_ignores_nan_when_fitting(device: torch.device) -> None:
    inp = torch.tensor(
        [[1.0, float("nan")], [3.0, 10.0], [5.0, 14.0]],
        device=device,
    )
    processor = Standardize().fit(TableTensor.from_tensor(inp))

    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    torch.testing.assert_close(
        processor.mean,
        torch.tensor([[3.0, 12.0]], device=device),
    )
    assert transformed[0, 1].isnan()
    torch.testing.assert_close(
        transformed[1, 0],
        torch.tensor(0.0, device=device),
    )


@withCUDA
def test_standardize_sparse_feature_uses_finite_count(
    device: torch.device,
) -> None:
    inp = torch.full((1000, 1), float("nan"), device=device)
    inp[:2, 0] = torch.tensor([100_000.0, 100_001.0], device=device)

    processor = Standardize().fit(TableTensor.from_tensor(inp))

    torch.testing.assert_close(
        processor.scale,
        torch.tensor([[0.5]], device=device),
    )
