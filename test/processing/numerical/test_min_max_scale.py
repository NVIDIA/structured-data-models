# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.processing import MinMaxScale
from sdm.testing import withCUDA


@withCUDA
def test_min_max_scale_maps_fitted_range_to_unit_interval(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]],
        dtype=torch.float64,
        device=device,
    )
    table = TableTensor.from_tensor(inp)

    processor = MinMaxScale().fit(table)
    out = processor.transform(table)

    expected = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        dtype=torch.float64,
        device=device,
    )
    torch.testing.assert_close(out.numerical, expected)
    torch.testing.assert_close(
        processor.inverse_transform(out).numerical,
        inp,
    )


@withCUDA
def test_min_max_scale_does_not_clip_values_outside_the_fitted_table(
    device: torch.device,
) -> None:
    context = torch.tensor(
        [[1.0, 5.0], [3.0, 5.0]],
        dtype=torch.float64,
        device=device,
    )
    query = torch.tensor(
        [[-1.0, 3.0], [7.0, 9.0]],
        dtype=torch.float64,
        device=device,
    )

    processor = MinMaxScale().fit(TableTensor.from_tensor(context))
    out = processor.transform(TableTensor.from_tensor(query))

    expected = torch.tensor(
        [[-1.0, -2.0], [3.0, 4.0]],
        dtype=torch.float64,
        device=device,
    )
    torch.testing.assert_close(out.numerical, expected)
    torch.testing.assert_close(
        processor.inverse_transform(out).numerical,
        query,
    )


@withCUDA
def test_min_max_scale_preserves_nan_and_inf(device: torch.device) -> None:
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

    out = MinMaxScale().fit_transform(TableTensor.from_tensor(inp))

    expected = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, float("nan")],
            [1.0, 0.0],
            [float("inf"), 0.0],
            [-float("inf"), float("inf")],
        ],
        dtype=torch.float64,
        device=device,
    )
    torch.testing.assert_close(out.numerical, expected, equal_nan=True)


@withCUDA
def test_min_max_scale_fits_leading_batches_independently(
    device: torch.device,
) -> None:
    context = torch.tensor(
        [[[1.0], [3.0]], [[10.0], [14.0]]],
        dtype=torch.float64,
        device=device,
    )
    query = torch.tensor(
        [[[2.0]], [[12.0]]],
        dtype=torch.float64,
        device=device,
    )

    processor = MinMaxScale().fit(TableTensor.from_tensor(context))
    out = processor.transform(TableTensor.from_tensor(query))

    expected = torch.full((2, 1, 1), 0.5, dtype=torch.float64, device=device)
    torch.testing.assert_close(out.numerical, expected)


@withCUDA
def test_min_max_scale_all_nonfinite_fit_yields_nan(
    device: torch.device,
) -> None:
    fit = torch.tensor(
        [[float("nan")], [float("inf")], [-float("inf")]],
        dtype=torch.float64,
        device=device,
    )
    query = torch.tensor([[1.0], [2.0]], dtype=torch.float64, device=device)

    out = (
        MinMaxScale()
        .fit(TableTensor.from_tensor(fit))
        .transform(TableTensor.from_tensor(query))
    )

    assert out.numerical.isnan().all()
