# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.processing import MinMaxScale
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_min_max_scale_fitted_range(
    device: torch.device, dtype: torch.dtype
) -> None:
    reference = torch.tensor(
        [[1.0, 5.0], [3.0, 5.0]], device=device, dtype=dtype
    )
    processor = MinMaxScale()
    out = processor.fit_transform(TableTensor.from_tensor(reference))
    torch.testing.assert_close(
        out.numerical, reference.new_tensor([[0.0, 0.0], [1.0, 0.0]])
    )

    query = reference.new_tensor([[-1.0, 3.0], [5.0, 7.0]])
    out = processor.transform(TableTensor.from_tensor(query))
    torch.testing.assert_close(
        out.numerical, query.new_tensor([[-1.0, -2.0], [2.0, 2.0]])
    )
    torch.testing.assert_close(
        processor.inverse_transform(out).numerical, query
    )


@withCUDA
def test_min_max_scale_batched_statistics(device: torch.device) -> None:
    reference = torch.tensor(
        [[[[1.0], [3.0]], [[10.0], [18.0]]]], device=device
    )
    query = torch.tensor([[[[2.0]], [[16.0]]]], device=device)
    processor = MinMaxScale().fit(TableTensor.from_tensor(reference))
    out = processor.transform(TableTensor.from_tensor(query))
    torch.testing.assert_close(
        out.numerical, query.new_tensor([[[[0.5]], [[0.75]]]])
    )


@withCUDA
def test_min_max_scale_nonfinite_values(device: torch.device) -> None:
    x = torch.tensor(
        [
            [1.0, torch.nan],
            [3.0, torch.inf],
            [torch.inf, -torch.inf],
            [-torch.inf, torch.nan],
            [torch.nan, torch.nan],
        ],
        device=device,
    )
    out = MinMaxScale().fit_transform(TableTensor.from_tensor(x))
    expected = x.clone()
    expected[:2, 0] = x.new_tensor([0.0, 1.0])
    expected[:, 1] = torch.nan
    torch.testing.assert_close(out.numerical, expected, equal_nan=True)
