# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import inf, nan

from sdm import TableTensor
from sdm.processing import Interpolate
from sdm.testing import withCUDA


@withCUDA
def test_interpolate(device: torch.device) -> None:
    context = TableTensor.from_tensor(
        torch.tensor([[nan], [1.0], [nan], [9.0], [nan]], device=device),
    )
    query = TableTensor.from_tensor(
        torch.tensor([[nan], [nan], [13.0], [nan]], device=device),
    )

    processor = Interpolate().fit(context)

    processor._offset = torch.tensor([0], device=device)
    torch.testing.assert_close(
        processor.transform(context).numerical,
        context.numerical.new_tensor([[1], [1], [5], [9], [9]]),
    )
    processor._offset = torch.tensor([5], device=device)
    torch.testing.assert_close(
        processor.transform(query).numerical,
        query.numerical.new_tensor([[11], [12], [13], [13]]),
    )


@withCUDA
def test_interpolate_with_leading_nan(device: torch.device) -> None:
    context = TableTensor.from_tensor(
        torch.tensor(
            [[nan, nan], [1.0, nan], [nan, nan], [5.0, nan]],
            device=device,
        ),
    )
    processor = Interpolate(fill_value=-1.0, fill_leading=False).fit(context)
    processor._offset = torch.tensor([0], device=device)

    torch.testing.assert_close(
        processor.transform(context).numerical,
        context.numerical.new_tensor(
            [[nan, -1.0], [1.0, -1.0], [3.0, -1.0], [5.0, -1.0]]
        ),
        equal_nan=True,
    )


@withCUDA
def test_interpolate_inf(device: torch.device) -> None:
    context = TableTensor.from_tensor(
        torch.tensor([[2], [nan], [nan], [inf], [nan]], device=device),
    )
    query = TableTensor.from_tensor(
        torch.tensor([[nan], [13.0], [nan]], device=device),
    )

    processor = Interpolate().fit(context)

    processor._offset = torch.tensor([0], device=device)
    torch.testing.assert_close(
        processor.transform(context).numerical,
        context.numerical.new_tensor([[2], [inf], [inf], [inf], [inf]]),
    )
    processor._offset = torch.tensor([5], device=device)
    torch.testing.assert_close(
        processor.transform(query).numerical,
        query.numerical.new_tensor([[inf], [13], [13]]),
    )
