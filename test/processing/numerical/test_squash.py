# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from sdm import TableTensor
from sdm.processing import SquashTransform
from sdm.testing import withCUDA

NAN = float("nan")
INF = float("inf")


def _make_input(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [-4.0, 2.0, 0.0, 1.0],
            [-1.0, 2.0, 5.0, NAN],
            [0.0, 2.0, 5.0, 3.0],
            [0.5, 2.0, 5.0, INF],
            [1.0, 2.0, 5.0, 4.0],
            [2.0, 2.0, 5.0, -INF],
            [3.0, 2.0, 5.0, 2.0],
            [10.0, 2.0, 5.0, NAN],
            [20.0, 2.0, 9.0, 8.0],
        ],
        device=device,
    )


def _reference(
    context: np.ndarray,
    query: np.ndarray,
    bound: float,
    quantile_range: tuple[float, float],
) -> np.ndarray:
    finite = np.where(np.isinf(context), np.nan, context)
    column_min = np.nanmin(finite, axis=0)
    column_max = np.nanmax(finite, axis=0)
    lower, median, upper = np.nanpercentile(
        finite, [quantile_range[0], 50.0, quantile_range[1]], axis=0
    )
    zero = column_max == column_min
    minmax = (lower == upper) & ~zero
    robust = ~(minmax | zero)
    infinite = np.sign(query) * np.isinf(query)
    out = np.where(np.isinf(query), np.nan, query)
    out[:, robust] = (out[:, robust] - median[robust]) / (
        upper[robust] - lower[robust]
    )
    tiny = np.finfo(query.dtype).tiny
    out[:, minmax] = (out[:, minmax] - median[minmax]) * (
        2.0 / (column_max[minmax] - column_min[minmax] + tiny)
    )
    zeroed = out[:, zero]
    out[:, zero] = np.where(np.isfinite(zeroed), 0.0, zeroed)
    out = out / np.sqrt(1 + (out / bound) ** 2)
    return np.where(infinite == 0, out, infinite * bound)


@withCUDA
def test_squash_transform_matches_reference(device: torch.device) -> None:
    inp = _make_input(device)
    query = torch.tensor(
        [[-7.0, 2.0, 5.0, NAN], [4.0, 9.0, 7.0, INF], [0.0, NAN, -INF, 2.0]],
        device=device,
    )
    processor = SquashTransform().fit(TableTensor.from_tensor(inp))
    out = processor.transform(TableTensor.from_tensor(inp)).numerical
    query_out = processor.transform(TableTensor.from_tensor(query)).numerical
    expected = _reference(inp.cpu().numpy(), inp.cpu().numpy(), 3.0, (25, 75))
    torch.testing.assert_close(
        out.cpu(), torch.from_numpy(expected), equal_nan=True
    )
    query_expected = _reference(
        inp.cpu().numpy(), query.cpu().numpy(), 3.0, (25, 75)
    )
    torch.testing.assert_close(
        query_out.cpu(), torch.from_numpy(query_expected), equal_nan=True
    )
    assert out.dtype == inp.dtype
    assert out.device == inp.device
    assert (out.nan_to_num().abs() <= 3.0).all()
    assert torch.equal(out.isnan(), inp.isnan())
    assert torch.equal(out[inp == INF], torch.full((1,), 3.0, device=device))
    assert torch.equal(out[inp == -INF], torch.full((1,), -3.0, device=device))
    assert torch.equal(out[:, 1], torch.zeros(inp.size(0), device=device))
    assert torch.equal(query_out[:2, 1], torch.zeros(2, device=device))


@withCUDA
def test_squash_transform_respects_custom_arguments(
    device: torch.device,
) -> None:
    inp = _make_input(device)
    processor = SquashTransform(
        max_absolute_value=1.5, quantile_range=(10.0, 90.0)
    ).fit(TableTensor.from_tensor(inp))
    out = processor.transform(TableTensor.from_tensor(inp)).numerical
    expected = _reference(
        inp.cpu().numpy(), inp.cpu().numpy(), 1.5, (10.0, 90.0)
    )
    torch.testing.assert_close(
        out.cpu(), torch.from_numpy(expected), equal_nan=True
    )
    assert (out.nan_to_num().abs() <= 1.5).all()
    assert repr(SquashTransform()) == "SquashTransform()"
    assert repr(
        SquashTransform(max_absolute_value=1.5, quantile_range=(10.0, 90.0))
    ) == (
        "SquashTransform(max_absolute_value=1.5, quantile_range=(10.0, 90.0))"
    )


@withCUDA
def test_squash_transform_fits_leading_batches_independently(
    device: torch.device,
) -> None:
    inp = _make_input(device)
    context = torch.stack([inp, inp.flip(0) * 3 - 1])
    query = torch.stack([inp[:3] + 1, inp[3:6] - 1])
    processor = SquashTransform().fit(TableTensor.from_tensor(context))
    actual = processor.transform(TableTensor.from_tensor(query)).numerical
    expected = []
    for batch in range(context.size(0)):
        independent = SquashTransform().fit(
            TableTensor.from_tensor(context[batch])
        )
        expected.append(
            independent.transform(
                TableTensor.from_tensor(query[batch])
            ).numerical
        )
    torch.testing.assert_close(actual, torch.stack(expected), equal_nan=True)
