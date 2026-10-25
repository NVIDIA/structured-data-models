# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.processing import WarpTails


def test_warp_tails_warps_columns_and_preserves_order() -> None:
    numerical = torch.linspace(-2.0, 2.0, 257, dtype=torch.float64)[
        :, None
    ].expand(-1, 8)
    table = TableTensor.from_tensor(numerical)

    output = WarpTails(probability=1.0).fit_transform(table).numerical

    assert output.dtype == numerical.dtype
    assert (output != numerical).any(dim=-2).all()
    assert output[128].eq(0).all()
    assert output.diff(dim=-2).ge(0).all()
