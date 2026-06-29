import pytest
import torch
from sdm.processing import SigmaClip


def test_sigma_clip_preserves_nan_positions_and_ignores_nan_in_fit() -> None:
    input = torch.tensor(
        [
            [1.0, torch.nan],
            [3.0, 5.0],
            [torch.nan, 7.0],
        ],
        dtype=torch.float64,
    )

    processor = SigmaClip().fit(input)
    transformed = processor.transform(input)

    assert torch.allclose(
        processor.mean,
        torch.tensor([2.0, 6.0], dtype=torch.float64),
    )
    assert torch.allclose(
        processor.std,
        torch.tensor([2.0**0.5, 2.0**0.5], dtype=torch.float64),
    )
    assert torch.equal(torch.isnan(transformed), torch.isnan(input))
    assert torch.isfinite(transformed[~torch.isnan(transformed)]).all()


def test_sigma_clip_constant_columns_use_minimum_std() -> None:
    input = torch.full((4, 2), 3.0)

    processor = SigmaClip().fit(input)
    transformed = processor.transform(input)

    assert torch.equal(processor.mean, torch.full((2,), 3.0))
    assert torch.equal(processor.std, torch.full((2,), 1e-6))
    assert torch.allclose(processor.lower_bound, torch.full((2,), 2.999996))
    assert torch.allclose(processor.upper_bound, torch.full((2,), 3.000004))
    assert torch.equal(transformed, input)


def test_sigma_clip_single_finite_value_uses_minimum_std() -> None:
    input = torch.tensor(
        [
            [torch.nan, 1.0],
            [torch.nan, torch.nan],
            [torch.nan, torch.nan],
        ]
    )

    processor = SigmaClip().fit(input)
    transformed = processor.transform(input)

    assert torch.isnan(processor.mean[0])
    assert torch.equal(processor.mean[1:], torch.tensor([1.0]))
    assert torch.equal(processor.std[1:], torch.tensor([1e-6]))
    assert torch.equal(torch.isnan(transformed), torch.isnan(input))
    assert torch.equal(
        transformed[~torch.isnan(input)],
        input[~torch.isnan(input)],
    )
    future = torch.tensor([[10.0, 1.0]])
    assert torch.equal(processor.transform(future), future)


def test_sigma_clip_two_stage_outlier_behavior() -> None:
    input = torch.tensor([[0.0], [1.0], [2.0], [100.0]], dtype=torch.float64)

    processor = SigmaClip(threshold=1.0).fit(input)
    transformed = processor.transform(input)

    assert torch.allclose(
        processor.mean, torch.tensor([1.0], dtype=torch.float64)
    )
    assert torch.allclose(
        processor.std, torch.tensor([1.0], dtype=torch.float64)
    )
    assert torch.allclose(
        processor.lower_bound,
        torch.tensor([0.0], dtype=torch.float64),
    )
    assert torch.allclose(
        processor.upper_bound,
        torch.tensor([2.0], dtype=torch.float64),
    )
    assert transformed[-1, 0] < input[-1, 0]
    assert torch.allclose(
        transformed[-1, 0],
        torch.log1p(torch.tensor(100.0, dtype=torch.float64)) + 2.0,
    )


def test_sigma_clip_matches_tabicl_reference_values() -> None:
    input = torch.tensor(
        [
            [-8.0, 1.0],
            [-1.0, torch.nan],
            [0.0, 3.0],
            [1.0, 4.0],
            [20.0, 5.0],
        ],
        dtype=torch.float64,
    )

    processor = SigmaClip(threshold=1.5).fit(input)
    transformed = processor.transform(input)

    assert torch.allclose(
        processor.mean,
        torch.tensor([-2.0, 3.25], dtype=torch.float64),
    )
    assert torch.allclose(
        processor.std,
        torch.tensor([4.082482904639, 1.70782512766], dtype=torch.float64),
    )
    assert torch.allclose(
        processor.lower_bound,
        torch.tensor([-8.123724356958, 0.68826230851], dtype=torch.float64),
    )
    assert torch.allclose(
        processor.upper_bound,
        torch.tensor([4.123724356958, 5.81173769149], dtype=torch.float64),
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
            dtype=torch.float64,
        ),
        equal_nan=True,
    )


def test_sigma_clip_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        SigmaClip(threshold=0.0)
