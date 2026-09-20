# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
def test_standardize(device: torch.device) -> None:
    inp = torch.tensor(
        [
            [1.0, 2.0, float("nan")],
            [3.0, 6.0, float("inf")],
            [float("nan"), float("inf"), -float("inf")],
        ],
        device=device,
    )

    processor = Standardize()
    processor.fit(TableTensor.from_tensor(inp))

    expected_mean = torch.tensor(
        [[2.0, 4.0, 0.0]],
        dtype=torch.float64,
        device=device,
    )
    expected_scale = torch.tensor(
        [[1.0, 2.0, 1.0]],
        dtype=torch.float64,
        device=device,
    )

    torch.testing.assert_close(processor.mean, expected_mean)
    torch.testing.assert_close(processor.scale, expected_scale)
    out = processor.transform(TableTensor.from_tensor(inp))

    expected = [
        [-1.0, -1.0, float("nan")],
        [1.0, 1.0, float("inf")],
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

    assert torch.equal(
        processor.scale,
        torch.ones((1, 2), dtype=torch.float64, device=device),
    )
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
        torch.tensor(
            [[[2.0]], [[12.0]]],
            dtype=torch.float64,
            device=device,
        ),
    )
    assert torch.equal(
        processor.scale,
        torch.tensor(
            [[[1.0]], [[2.0]]],
            dtype=torch.float64,
            device=device,
        ),
    )
    assert torch.equal(out.numerical, torch.full_like(query, 2.0))
    assert torch.equal(
        processor.inverse_transform(out).numerical,
        query,
    )


@withCUDA
def test_standardize_computes_in_float64(device: torch.device) -> None:
    inp = torch.tensor(
        [[1e8], [1e8 + 8], [1e8 + 8]],
        dtype=torch.float32,
        device=device,
    )

    output = Standardize().fit_transform(TableTensor.from_tensor(inp))

    expected = torch.tensor(
        [[-(2**0.5)], [2**-0.5], [2**-0.5]],
        dtype=torch.float32,
        device=device,
    )
    assert output.numerical.dtype == inp.dtype
    torch.testing.assert_close(output.numerical, expected)
