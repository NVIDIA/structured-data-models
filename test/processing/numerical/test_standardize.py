# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.processing import Standardize
from sdm.testing import withCUDA


@withCUDA
def test_standardize_with_state(device: torch.device) -> None:
    processor = Standardize().with_state(mean=2.0, scale=3.0).to(device)
    data = torch.randn(2, 3, 2, device=device)
    table = TableTensor.from_tensor(data)
    expected = (data - 2.0) / 3.0

    torch.testing.assert_close(processor.transform(table).numerical, expected)
    processor.fit(table)
    torch.testing.assert_close(processor.transform(table).numerical, expected)
    out = processor.fit_transform(table)
    torch.testing.assert_close(out.numerical, expected)


@pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile is not supported by this Python/PyTorch build",
)
def test_standardize_compile_fit() -> None:
    def fit(x: torch.Tensor) -> tuple[torch.Tensor, Standardize]:
        processor = Standardize()
        table = TableTensor.from_tensor(x)
        return processor.fit_transform(table).numerical, processor

    context = torch.tensor([[1.0, 2.0], [3.0, torch.nan], [5.0, 6.0]])
    compiled_fit = torch.compile(fit, backend="eager", fullgraph=True)
    expected, eager = fit(context)
    actual, fitted = compiled_fit(context)
    torch.testing.assert_close(actual, expected, equal_nan=True)

    restored = Standardize()
    restored.load_state_dict(fitted.state_dict())
    assert restored.is_fitted
    query = TableTensor.from_tensor(torch.tensor([[7.0, 8.0]]))
    torch.testing.assert_close(
        restored.transform(query).numerical, eager.transform(query).numerical
    )


@pytest.mark.parametrize("name", ["mean", "scale", "_fitted_state"])
def test_standardize_preserves_nonpersistent_buffers(name: str) -> None:
    processor = Standardize()
    processor.register_buffer(name, getattr(processor, name), persistent=False)
    processor.fit(TableTensor.from_tensor(torch.randn(3, 2)))
    assert name not in processor.state_dict()


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
def test_standardize_ignores_non_finite_values_in_many_rows(
    device: torch.device,
) -> None:
    inp = torch.randn(2, 600, 3, device=device)
    inp[inp > 1.0] = float("nan")
    inp[inp < -1.5] = float("inf")

    out = Standardize().fit_transform(TableTensor.from_tensor(inp))

    finite = inp.masked_fill(~inp.isfinite(), float("nan"))
    mean = finite.nanmean(-2, keepdim=True)
    std = (finite - mean).square().nanmean(-2, keepdim=True).sqrt()
    torch.testing.assert_close(
        out.numerical,
        (inp - mean) / std,
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
