# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.prior import LogUniform
from sdm.prior.tabular.transform import MaskNearEmpty


def test_mask_near_empty_keeps_sparse_columns_in_overlapping_rows() -> None:
    num_rows, num_observed = 256, 16
    values = torch.zeros(num_rows, 8)
    diagonal = torch.arange(8)
    values[diagonal, diagonal] = 1
    x = TableTensor(numerical=values)
    y = TableTensor.from_tensor(torch.zeros(num_rows, 1))
    rate = num_observed / num_rows

    output, output_y = MaskNearEmpty(
        columns=(1.0, 1.0),
        observed=LogUniform(rate, rate),
    )(x=x, y=y)

    missing = output.numerical.isnan()
    assert missing.sum(dim=0).eq(num_rows - num_observed).all()
    observed = ~missing
    overlap = observed.mT.int() @ observed.int()
    overlap.fill_diagonal_(-1)
    assert overlap.amax(dim=-1).ge(num_observed - 2).all()
    assert all(
        output.numerical[:, i][observed[:, i]].unique().numel() == 2
        for i in range(8)
    )
    assert output.numerical[~missing].equal(x.numerical[~missing])
    assert output_y.numerical.equal(y.numerical)
