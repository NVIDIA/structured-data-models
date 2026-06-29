import pytest
import torch
from sdm.processing import SigmaClip
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
def test_sigma_clip_preserves_nan_positions_and_ignores_nan_in_fit(
    device: torch.device,
    dtype: torch.dtype | None,
) -> None:
    input = torch.tensor(
        [
            [1.0, torch.nan, 3.0, torch.nan],
            [3.0, 5.0, 3.0, torch.nan],
            [torch.nan, 7.0, 3.0, torch.nan],
        ],
        dtype=dtype,
        device=device,
    )

    processor = SigmaClip().fit(input)
    transformed = processor.transform(input)

    assert torch.allclose(
        processor._mean[:3],
        torch.tensor([2.0, 6.0, 3.0], dtype=dtype, device=device),
    )
    assert torch.isnan(processor._mean[3])
    # Constant and all-NaN columns fall back to the minimum std.
    assert torch.allclose(
        processor._std,
        torch.tensor(
            [2.0**0.5, 2.0**0.5, 1e-6, 1e-6], dtype=dtype, device=device
        ),
    )
    # The constant column gets a tiny band of ``mean ± threshold * min_std``.
    assert torch.allclose(
        processor.lower_bound[2],
        torch.tensor(3.0 - 4e-6, dtype=dtype, device=device),
    )
    assert torch.allclose(
        processor.upper_bound[2],
        torch.tensor(3.0 + 4e-6, dtype=dtype, device=device),
    )
    # All-NaN columns get a NaN mean and ±inf bounds, so values pass through.
    assert torch.isneginf(processor.lower_bound[3])
    assert torch.isposinf(processor.upper_bound[3])
    assert torch.equal(torch.isnan(transformed), torch.isnan(input))
    assert torch.isfinite(transformed[~torch.isnan(transformed)]).all()


@withCUDA
def test_sigma_clip_two_stage_outlier_behavior(
    device: torch.device,
) -> None:
    dtype = torch.float64
    input = torch.tensor(
        [[0.0], [1.0], [2.0], [100.0]],
        dtype=dtype,
        device=device,
    )

    processor = SigmaClip(threshold=1.0).fit(input)
    transformed = processor.transform(input)

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
    assert transformed[-1, 0] < input[-1, 0]
    assert torch.allclose(
        transformed[-1, 0],
        torch.log1p(torch.tensor(100.0, dtype=dtype, device=device)) + 2.0,
    )


def test_sigma_clip_matches_tabicl_reference_values() -> None:
    dtype = torch.float64
    input = torch.tensor(
        [
            [-8.0, 1.0],
            [-1.0, torch.nan],
            [0.0, 3.0],
            [1.0, 4.0],
            [20.0, 5.0],
        ],
        dtype=dtype,
    )

    processor = SigmaClip(threshold=1.5).fit(input)
    transformed = processor.transform(input)

    assert torch.allclose(
        processor._mean,
        torch.tensor([-2.0, 3.25], dtype=dtype),
    )
    assert torch.allclose(
        processor._std,
        torch.tensor([4.082482904639, 1.70782512766], dtype=dtype),
    )
    assert torch.allclose(
        processor.lower_bound,
        torch.tensor([-8.123724356958, 0.68826230851], dtype=dtype),
    )
    assert torch.allclose(
        processor.upper_bound,
        torch.tensor([4.123724356958, 5.81173769149], dtype=dtype),
    )
    assert torch.allclose(
        transformed,
        torch.tensor(
            [
                [-8.0, 1.0],
                [-1.0, torch.nan],
                [0.0, 3.0],
                [1.0, 4.0],
                [7.168246794681, 5.0],
            ],
            dtype=dtype,
        ),
        equal_nan=True,
    )


def test_sigma_clip_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        SigmaClip(threshold=0.0)
