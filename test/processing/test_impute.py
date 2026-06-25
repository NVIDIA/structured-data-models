from __future__ import annotations

import torch
from schemafm.processing import MeanImpute


def test_mean_impute_replaces_nan_with_column_mean() -> None:
    input = torch.tensor(
        [
            [1.0, torch.nan],
            [3.0, 5.0],
            [torch.nan, 7.0],
        ],
        dtype=torch.float64,
    )

    processor = MeanImpute().fit(input)
    transformed = processor.transform(input)

    assert torch.allclose(
        processor.mean,
        torch.tensor([2.0, 6.0], dtype=torch.float64),
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
        ),
    )


def test_mean_impute_no_nan_keeps_values() -> None:
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)

    processor = MeanImpute().fit(input)

    assert torch.equal(processor.transform(input), input)


def test_mean_impute_all_nan_column_uses_fill_value_and_keeps_shape() -> None:
    input = torch.tensor(
        [
            [torch.nan, 1.0],
            [torch.nan, 3.0],
        ]
    )

    processor = MeanImpute(fill_value=-5.0).fit(input)
    transformed = processor.transform(input)

    assert transformed.shape == input.shape
    assert torch.equal(processor.mean, torch.tensor([-5.0, 2.0]))
    assert torch.equal(
        transformed,
        torch.tensor(
            [
                [-5.0, 1.0],
                [-5.0, 3.0],
            ]
        ),
    )


def test_mean_impute_preserves_floating_dtype() -> None:
    input = torch.tensor([[1.0], [torch.nan]], dtype=torch.float32)

    transformed = MeanImpute().fit(input).transform(input)

    assert transformed.dtype == torch.float32
    assert torch.equal(
        transformed, torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    )
