# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        processor.lower_bound,
        torch.tensor([[0.0]], dtype=dtype, device=device),
    )
    assert torch.allclose(
        processor.upper_bound,
        torch.tensor([[2.0]], dtype=dtype, device=device),
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
        processor.lower_bound,
        torch.tensor([[-8.123724356958, 0.628291754874]], dtype=dtype),
    )
    assert torch.allclose(
        processor.upper_bound,
        torch.tensor([[4.123724356958, 5.371708245126]], dtype=dtype),
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


@withCUDA
@pytest.mark.parametrize("threshold", [0.1, 1.5])
def test_clip_sigma_preserves_strided_inputs_and_promotes_query_dtype(
    device: torch.device, threshold: float
) -> None:
    context = (
        torch.tensor(
            [[-8.0, 1.0], [-1.0, 2.0], [0.0, 3.0], [1.0, 4.0], [20.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        .T.contiguous()
        .T
    )
    original = context.clone()
    processor = ClipSigma(threshold=threshold).fit(
        TableTensor(numerical=context)
    )
    query = (
        torch.tensor(
            [
                [-100.0, float("nan")],
                [float("inf"), -float("inf")],
                [1.0, 100.0],
            ],
            device=device,
        )
        .T.contiguous()
        .T
    )
    before = query.clone()
    lower = processor.lower_bound.clone()
    upper = processor.upper_bound.clone()
    expected = torch.minimum(
        query.abs().log1p() + upper,
        torch.maximum(-query.abs().log1p() + lower, query),
    )

    first = processor.transform(TableTensor(numerical=query)).numerical
    second = processor.transform(TableTensor(numerical=query)).numerical

    torch.testing.assert_close(first, expected, equal_nan=True)
    torch.testing.assert_close(second, expected, equal_nan=True)
    torch.testing.assert_close(context, original)
    torch.testing.assert_close(query, before, equal_nan=True)
    torch.testing.assert_close(processor.lower_bound, lower)
    torch.testing.assert_close(processor.upper_bound, upper)


def test_clip_sigma_preserves_query_gradients() -> None:
    processor = ClipSigma(threshold=1.0).fit(
        TableTensor(numerical=torch.tensor([[0.0], [1.0], [2.0], [100.0]]))
    )
    query = torch.tensor([[-100.0], [1.0], [100.0]], requires_grad=True)
    processor.transform(
        TableTensor(numerical=query)
    ).numerical.sum().backward()
    torch.testing.assert_close(
        query.grad, query.new_tensor([[1 / 101], [1], [1 / 101]])
    )


@withCUDA
@pytest.mark.parametrize("num_rows", [1, 4])
def test_clip_sigma_fits_all_missing_columns(
    device: torch.device, num_rows: int
) -> None:
    values = torch.tensor(
        [[float("nan"), float("inf"), -float("inf")]], device=device
    ).expand(num_rows, -1)
    out = ClipSigma().fit_transform(TableTensor(numerical=values))
    torch.testing.assert_close(out.numerical, values, equal_nan=True)


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
    transformed = processor.transform(
        TableTensor.from_tensor(
            torch.tensor(
                [[[-100.0], [100.0]], [[-200.0], [200.0]]],
                dtype=torch.float64,
                device=device,
            )
        )
    ).numerical

    torch.testing.assert_close(
        transformed,
        torch.tensor(
            [
                [[-4.61512051684126], [6.61512051684126]],
                [[4.696695091940924], [19.303304908059076]],
            ],
            dtype=torch.float64,
            device=device,
        ),
    )


@withCUDA
@pytest.mark.parametrize(
    "missing", [float("nan"), float("inf"), -float("inf")]
)
def test_clip_sigma_preserves_nonfinite(
    device: torch.device,
    missing: float,
) -> None:
    inp = torch.tensor(
        [
            [0.0, missing],
            [1.0, 10.0],
            [2.0, 12.0],
            [100.0, 14.0],
        ],
        device=device,
    )

    processor = ClipSigma(threshold=1.0)
    out = processor.fit_transform(TableTensor.from_tensor(inp))

    torch.testing.assert_close(
        out.numerical,
        torch.tensor(
            [
                [0.0, missing],
                [1.0, 10.0],
                [2.0, 12.0],
                [6.6151205, 14.0],
            ],
            device=device,
        ),
        equal_nan=True,
    )
