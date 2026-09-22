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
    expected = torch.tensor(
        [
            [-((3.0 / 2.0) ** 0.5), 0.0, -((3.0 / 2.0) ** 0.5)],
            [0.0, 0.0, 0.0],
            [(3.0 / 2.0) ** 0.5, 0.0, (3.0 / 2.0) ** 0.5],
        ],
        dtype=torch.float64,
        device=device,
    )

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

    assert torch.equal(out.numerical, torch.full_like(query, 2.0))
    assert torch.equal(
        processor.inverse_transform(out).numerical,
        query,
    )


def test_standardize_computes_in_float64() -> None:
    inp = torch.tensor(
        [[1e8], [1e8 + 8], [1e8 + 8]],
        dtype=torch.float32,
    )

    output = Standardize().fit_transform(TableTensor.from_tensor(inp))

    expected = torch.tensor(
        [[-(2**0.5)], [2**-0.5], [2**-0.5]],
        dtype=torch.float32,
    )
    assert output.numerical.dtype == inp.dtype
    torch.testing.assert_close(output.numerical, expected)
