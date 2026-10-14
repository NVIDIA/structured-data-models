# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.processing import Detrend
from sdm.testing import withCUDA


@withCUDA
def test_detrend(device: torch.device) -> None:
    context = TableTensor.from_tensor(
        torch.tensor([[1.0], [5.0], [9.0]], device=device),
    )
    query = TableTensor.from_tensor(
        torch.tensor([[20.0], [22.0]], device=device),
    )

    processor = Detrend().fit(context)

    processor._offset = torch.tensor([0], device=device)
    torch.testing.assert_close(
        processor.transform(context).numerical,
        torch.zeros_like(context.numerical),
    )
    processor._offset = torch.tensor([3], device=device)
    torch.testing.assert_close(
        processor.transform(query).numerical,
        query.numerical.new_tensor([[7.0], [5.0]]),
    )


@withCUDA
def test_detrend_threshold(device: torch.device) -> None:
    context = TableTensor.from_tensor(
        torch.tensor([[0.0], [10.0], [0.0], [10.0]], device=device),
    )
    query = TableTensor.from_tensor(
        torch.tensor([[1.0], [2.0]], device=device),
    )

    processor = Detrend(threshold=0.5).fit(context)

    processor._offset = torch.tensor([0], device=device)
    torch.testing.assert_close(
        processor.transform(context).numerical,
        context.numerical,
    )
    processor._offset = torch.tensor([4], device=device)
    torch.testing.assert_close(
        processor.transform(query).numerical,
        query.numerical,
    )


@withCUDA
def test_detrend_nan(device: torch.device) -> None:
    context = TableTensor.from_tensor(
        torch.tensor([[1.0], [torch.nan], [9.0]], device=device),
    )
    query = TableTensor.from_tensor(
        torch.tensor([[13.0]], device=device),
    )

    processor = Detrend().fit(context)

    processor._offset = torch.tensor([0], device=device)
    torch.testing.assert_close(
        processor.transform(context).numerical,
        context.numerical.new_tensor([[0.0], [torch.nan], [0.0]]),
        equal_nan=True,
    )
    processor._offset = torch.tensor([3], device=device)
    torch.testing.assert_close(
        processor.transform(query).numerical,
        torch.zeros_like(query.numerical),
    )
