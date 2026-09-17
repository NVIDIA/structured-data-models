# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.processing import RobustScale
from sdm.testing import withCUDA


@withCUDA
def test_robust_scale_centers_and_scales_quantile_range(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
            [4.0, 5.0],
            [5.0, 5.0],
        ],
        dtype=torch.float64,
        device=device,
    )
    table = TableTensor.from_tensor(inp)

    processor = RobustScale().fit(table)
    out = processor.transform(table)

    expected = torch.cat(
        [(inp[:, :1] - 3.0) / 2.0, torch.zeros_like(inp[:, :1])],
        dim=-1,
    )
    torch.testing.assert_close(out.numerical, expected)
    torch.testing.assert_close(
        processor.inverse_transform(out).numerical,
        inp,
    )


@withCUDA
def test_robust_scale_uses_half_span_when_quantiles_coincide(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [10.0]],
        dtype=torch.float64,
        device=device,
    )

    processor = RobustScale().fit(TableTensor.from_tensor(inp))
    out = processor.transform(TableTensor.from_tensor(inp))
    expected = (inp - 1.0) / 4.5

    torch.testing.assert_close(out.numerical, expected)
    torch.testing.assert_close(
        processor.inverse_transform(out).numerical,
        inp,
    )


@withCUDA
def test_robust_scale_preserves_nan_and_inf(device: torch.device) -> None:
    inp = torch.tensor(
        [
            [1.0, 5.0],
            [2.0, float("nan")],
            [3.0, 5.0],
            [float("inf"), 5.0],
            [-float("inf"), float("inf")],
        ],
        dtype=torch.float64,
        device=device,
    )

    out = RobustScale().fit_transform(TableTensor.from_tensor(inp))

    expected = torch.tensor(
        [
            [-1.0, 0.0],
            [0.0, float("nan")],
            [1.0, 0.0],
            [float("inf"), 0.0],
            [-float("inf"), float("inf")],
        ],
        dtype=torch.float64,
        device=device,
    )
    torch.testing.assert_close(out.numerical, expected, equal_nan=True)


@withCUDA
def test_robust_scale_fits_leading_batches_independently(
    device: torch.device,
) -> None:
    context = torch.tensor(
        [[[1.0], [3.0], [5.0]], [[10.0], [14.0], [18.0]]],
        dtype=torch.float64,
        device=device,
    )
    query = torch.tensor(
        [[[5.0], [7.0]], [[18.0], [22.0]]],
        dtype=torch.float64,
        device=device,
    )

    processor = RobustScale().fit(TableTensor.from_tensor(context))
    out = processor.transform(TableTensor.from_tensor(query))

    expected = torch.tensor(
        [[[1.0], [2.0]], [[1.0], [2.0]]],
        dtype=torch.float64,
        device=device,
    )
    torch.testing.assert_close(out.numerical, expected)


@withCUDA
def test_robust_scale_uses_quantile_range(device: torch.device) -> None:
    inp = torch.arange(1, 5, dtype=torch.float64, device=device).unsqueeze(-1)

    out = RobustScale(quantile_range=(0.0, 100.0)).fit_transform(
        TableTensor.from_tensor(inp)
    )

    torch.testing.assert_close(out.numerical, (inp - 2.5) / 3.0)


@withCUDA
def test_robust_scale_all_nonfinite_fit_yields_nan(
    device: torch.device,
) -> None:
    fit = torch.tensor(
        [[float("nan")], [float("inf")], [-float("inf")]],
        dtype=torch.float64,
        device=device,
    )
    query = torch.tensor([[1.0], [2.0]], dtype=torch.float64, device=device)

    out = (
        RobustScale()
        .fit(TableTensor.from_tensor(fit))
        .transform(TableTensor.from_tensor(query))
    )

    assert out.numerical.isnan().all()


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_robust_scale_fits_half_precision(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    inp = torch.tensor(
        [[1.0], [3.0], [5.0]],
        dtype=dtype,
        device=device,
    )
    table = TableTensor.from_tensor(inp)
    out = RobustScale().fit_transform(table)

    assert out.numerical.dtype == dtype
    expected = torch.tensor([[-1.0], [0.0], [1.0]], device=device)
    torch.testing.assert_close(out.numerical.float(), expected)


def test_robust_scale_rejects_invalid_quantile_range() -> None:
    with pytest.raises(ValueError, match="quantile_range"):
        RobustScale(quantile_range=(75.0, 25.0))
    with pytest.raises(ValueError, match="quantile_range"):
        RobustScale(quantile_range=(-1.0, 75.0))
    with pytest.raises(ValueError, match="quantile_range"):
        RobustScale(quantile_range=(25.0, 101.0))
