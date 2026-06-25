from __future__ import annotations

import torch
from sdm.processing import StandardScale


def test_standard_scale_fit_transform_and_inverse_round_trip() -> None:
    input = torch.tensor(
        [
            [1.0, 2.0, 5.0],
            [3.0, 2.0, 7.0],
            [5.0, 2.0, 9.0],
        ],
        dtype=torch.float64,
    )

    processor = StandardScale().fit(input)
    expected_mean = torch.tensor([3.0, 2.0, 7.0], dtype=torch.float64)
    expected_scale = torch.tensor(
        [
            torch.sqrt(torch.tensor(8.0 / 3.0)),
            1.0,
            torch.sqrt(torch.tensor(8.0 / 3.0)),
        ],
        dtype=torch.float64,
    )
    expected = (input - expected_mean) / expected_scale

    assert torch.allclose(processor.mean, expected_mean)
    assert torch.allclose(processor.scale, expected_scale)
    transformed = processor.transform(input)
    assert torch.allclose(transformed, expected)
    assert torch.allclose(processor.inverse_transform(transformed), input)


def test_standard_scale_without_mean_or_std() -> None:
    input = torch.tensor([[1.0, 2.0], [3.0, 6.0]])

    processor = StandardScale(with_mean=False, with_std=False).fit(input)

    assert torch.equal(processor.mean, torch.zeros(2))
    assert torch.equal(processor.scale, torch.ones(2))
    assert torch.equal(processor.transform(input), input)


def test_standard_scale_nan_columns_follow_torch_reductions() -> None:
    input = torch.tensor(
        [
            [1.0, 1.0],
            [torch.nan, 3.0],
            [5.0, 5.0],
        ]
    )

    processor = StandardScale().fit(input)
    transformed = processor.transform(input)

    assert torch.isnan(processor.mean[0])
    assert torch.isnan(processor.scale[0])
    assert torch.isnan(transformed[:, 0]).all()
    assert torch.isfinite(transformed[:, 1]).all()


def test_standard_scale_single_sample_uses_unit_scale() -> None:
    input = torch.tensor([[42.0, -2.0]])

    processor = StandardScale().fit(input)
    transformed = processor.transform(input)

    assert torch.equal(processor.scale, torch.ones(2))
    assert torch.equal(transformed, torch.zeros_like(input))
    assert torch.equal(processor.inverse_transform(transformed), input)
