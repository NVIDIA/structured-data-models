import pytest
import torch

from sdm import TableTensor
from sdm.processing import ClipSigma
from sdm.testing import withCUDA


@withCUDA
def test_clip_sigma_two_stage_outlier_behavior(
    device: torch.device,
) -> None:
    dtype = torch.float64
    inp = torch.tensor(
        [[0.0], [1.0], [2.0], [100.0]],
        dtype=dtype,
        device=device,
    )

    processor = ClipSigma(threshold=1.0).fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.allclose(
        processor._mean,
        torch.tensor([1.0], dtype=dtype, device=device),
    )
    assert torch.allclose(
        processor._std,
        torch.tensor([1.0], dtype=dtype, device=device),
    )
    assert torch.allclose(
        processor.lower_bound,
        torch.tensor([0.0], dtype=dtype, device=device),
    )
    assert torch.allclose(
        processor.upper_bound,
        torch.tensor([2.0], dtype=dtype, device=device),
    )
    assert transformed[-1, 0] < inp[-1, 0]
    assert torch.allclose(
        transformed[-1, 0],
        torch.log1p(torch.tensor(100.0, dtype=dtype, device=device)) + 2.0,
    )


def test_clip_sigma_matches_tabicl_reference_values() -> None:
    dtype = torch.float64
    inp = torch.tensor(
        [
            [-8.0, 1.0],
            [-1.0, 2.0],
            [0.0, 3.0],
            [1.0, 4.0],
            [20.0, 5.0],
        ],
        dtype=dtype,
    )

    processor = ClipSigma(threshold=1.5).fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.allclose(
        processor._mean,
        torch.tensor([-2.0, 3.0], dtype=dtype),
    )
    assert torch.allclose(
        processor._std,
        torch.tensor([4.082482904639, 1.581138830084], dtype=dtype),
    )
    assert torch.allclose(
        processor.lower_bound,
        torch.tensor([-8.123724356958, 0.628291754874], dtype=dtype),
    )
    assert torch.allclose(
        processor.upper_bound,
        torch.tensor([4.123724356958, 5.371708245126], dtype=dtype),
    )
    assert torch.allclose(
        transformed,
        torch.tensor(
            [
                [-8.0, 1.0],
                [-1.0, 2.0],
                [0.0, 3.0],
                [1.0, 4.0],
                [7.168246794681, 5.0],
            ],
            dtype=dtype,
        ),
    )


def test_clip_sigma_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        ClipSigma(threshold=0.0)


@withCUDA
def test_clip_sigma_fits_leading_batches_independently(
    device: torch.device,
) -> None:
    context = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [100.0]],
            [[10.0], [12.0], [14.0], [200.0]],
        ],
        dtype=torch.float64,
        device=device,
    )

    processor = ClipSigma(threshold=1.0).fit(TableTensor.from_tensor(context))
    query = processor.transform(
        TableTensor.from_tensor(
            torch.tensor([[[1.0]], [[12.0]]], device=device)
        )
    ).numerical

    assert torch.equal(
        processor._mean,
        torch.tensor([[[1.0]], [[12.0]]], dtype=torch.float64, device=device),
    )
    assert torch.equal(
        processor._std,
        torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float64, device=device),
    )
    assert torch.equal(
        query,
        torch.tensor([[[1.0]], [[12.0]]], dtype=torch.float64, device=device),
    )
