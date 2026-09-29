# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.processing import ClipSigma
from sdm.testing import onlyCUDA, withCUDA


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


@withCUDA
@pytest.mark.parametrize("fit_transform", [False, True])
def test_clip_sigma_gradients(
    device: torch.device,
    fit_transform: bool,
) -> None:
    context = torch.tensor(
        [[1.0], [2.0], [4.0], [100.0]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    query = TableTensor.from_tensor(context.detach().clone())

    def transform(values: torch.Tensor) -> torch.Tensor:
        processor = ClipSigma(threshold=1.0)
        table = TableTensor.from_tensor(values)
        if fit_transform:
            return processor.fit_transform(table).numerical
        processor.fit(table)
        return processor.transform(query).numerical

    assert torch.autograd.gradcheck(transform, (context,))


@onlyCUDA
def test_clip_sigma_row_chunks_match_single_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = torch.randn(2, 50, 3, device="cuda", dtype=torch.float64)
    inp[:, ::7, 0] = float("nan")
    inp[:, 3, 1] = float("inf")
    processor = ClipSigma(threshold=1.0)
    processor.fit(TableTensor.from_tensor(inp))
    query = TableTensor.from_tensor(inp.float() * 3)

    # Passes run without autograd:
    with torch.inference_mode():
        expected = processor.transform(query)
        # Passes of a single row:
        monkeypatch.setenv("SDM_CHUNK_MEMORY_FRACTION", "1e-12")
        out = processor.transform(query)

    assert out.numerical.dtype == torch.float64
    torch.testing.assert_close(
        out.numerical,
        expected.numerical,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
