# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.processing import RankGaussian
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mid_ranks_and_interpolated_query(
    dtype: torch.dtype, device: torch.device
) -> None:
    context = torch.tensor(
        [[0.0], [1.0], [1.0], [2.0]], dtype=dtype, device=device
    )
    processor = RankGaussian().fit(TableTensor.from_tensor(context))
    probabilities = context.new_tensor([[0.125], [0.5], [0.5], [0.875]])
    torch.testing.assert_close(
        processor.transform(TableTensor.from_tensor(context)).numerical,
        torch.special.ndtri(probabilities),
    )
    query = context.new_tensor([[-100.0], [0.5], [1.5], [100.0], [torch.nan]])
    expected = context.new_tensor(
        [[0.125], [0.3125], [0.6875], [0.875], [torch.nan]]
    )
    torch.testing.assert_close(
        processor.transform(TableTensor.from_tensor(query)).numerical,
        torch.special.ndtri(expected),
        equal_nan=True,
    )


@withCUDA
def test_ties_at_endpoints_use_mid_ranks(device: torch.device) -> None:
    context = torch.tensor([[0.0], [0.0], [2.0], [2.0]], device=device)
    output = RankGaussian().fit_transform(TableTensor.from_tensor(context))
    probabilities = context.new_tensor([[0.25], [0.25], [0.75], [0.75]])
    torch.testing.assert_close(
        output.numerical, torch.special.ndtri(probabilities)
    )


@withCUDA
@pytest.mark.parametrize("n_rows", [1, 4])
def test_constants_and_missing_columns(
    n_rows: int, device: torch.device
) -> None:
    context = torch.tensor([[7.0, torch.nan]], device=device).expand(
        n_rows, -1
    )
    processor = RankGaussian().fit(TableTensor.from_tensor(context))
    query = context.new_tensor(
        [[7.0, 8.0], [torch.nan, torch.nan], [100.0, 3.0]]
    )
    expected = context.new_tensor(
        [[0.0, torch.nan], [torch.nan, torch.nan], [0.0, torch.nan]]
    )
    torch.testing.assert_close(
        processor.transform(TableTensor.from_tensor(query)).numerical,
        expected,
        equal_nan=True,
    )


@withCUDA
def test_nonfinite_context_does_not_change_observed_ranks(
    device: torch.device,
) -> None:
    context = torch.tensor(
        [[0.0], [1.0], [torch.nan], [1.0], [torch.inf], [-torch.inf], [2.0]],
        device=device,
    )
    processor = RankGaussian().fit(TableTensor.from_tensor(context))
    reference = RankGaussian().fit(
        TableTensor.from_tensor(context[[0, 1, 3, 6]])
    )
    query = TableTensor.from_tensor(context)
    output = processor.transform(query).numerical
    torch.testing.assert_close(
        output, reference.transform(query).numerical, equal_nan=True
    )
    assert torch.equal(output.isnan(), context.isnan())
    assert output[~context.isnan()].isfinite().all()


@withCUDA
def test_query_batch_does_not_change_fitted_ranks(
    device: torch.device,
) -> None:
    context = torch.arange(10.0, device=device).unsqueeze(-1)
    processor = RankGaussian().fit(TableTensor.from_tensor(context))
    query = context.new_tensor([[1.5], [torch.nan], [5.5]])
    expected = processor.transform(TableTensor.from_tensor(query)).numerical
    extended = torch.cat([query, context.new_tensor([[-1e12], [1e12]])])
    actual = processor.transform(TableTensor.from_tensor(extended)).numerical
    torch.testing.assert_close(actual[:3], expected, equal_nan=True)


@withCUDA
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_batched_missing_values_match_independent_columns(
    batch_shape: tuple[int, ...], device: torch.device
) -> None:
    context = torch.randn(*batch_shape, 17, 4, device=device)
    context[..., 1, 0] = torch.nan
    context[..., :3, 1] = torch.nan
    context[..., :, 2] = 2.0
    context[..., :, 3] = torch.nan
    query = torch.randn(*batch_shape, 7, 4, device=device)
    query[..., 0, 0] = torch.nan
    processor = RankGaussian().fit(TableTensor.from_tensor(context))
    output = processor.transform(TableTensor.from_tensor(query)).numerical
    for train, test, actual in zip(
        context.reshape(-1, 17, 4),
        query.reshape(-1, 7, 4),
        output.reshape(-1, 7, 4),
        strict=True,
    ):
        for column in range(4):
            reference = RankGaussian().fit(
                TableTensor.from_tensor(train[:, column : column + 1])
            )
            expected = reference.transform(
                TableTensor.from_tensor(test[:, column : column + 1])
            ).numerical
            torch.testing.assert_close(
                actual[:, column : column + 1], expected, equal_nan=True
            )
