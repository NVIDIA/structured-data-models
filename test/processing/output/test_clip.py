# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import EnsembleTable, TableTensor
from sdm.processing import ClipNonnegative
from sdm.testing import withCUDA


@withCUDA
def test_clip_nonnegative(device: torch.device) -> None:
    target = torch.tensor(
        [[[0.0, -1.0], [1.0, 1.0]], [[1.0, 0.0], [2.0, 2.0]]],
        device=device,
    ).repeat(2, 1, 1, 1)
    target[1, 1, 0, 0] = -1.0

    processor = ClipNonnegative()
    processor._target = EnsembleTable(
        groups=[TableTensor.from_tensor(target)],
        locations=[(0, 0), (0, 1)],
    )

    out = target.new_tensor([[[-2.0, 1.0, -2.0, 1.0]]]).expand(2, 1, -1)
    out = TableTensor.from_tensor(out)
    expected = target.new_tensor(
        [
            [[0.0, 1.0, -2.0, 1.0]],
            [[-2.0, 1.0, 0.0, 1.0]],
        ]
    )
    torch.testing.assert_close(processor.transform(out).numerical, expected)
