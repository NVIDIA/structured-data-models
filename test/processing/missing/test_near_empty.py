# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import Stype, TableTensor
from sdm.processing import MaskNearEmpty
from sdm.testing import withCUDA


@withCUDA
def test_mask_near_empty_keeps_sparse_columns_in_overlapping_rows(
    device: torch.device,
) -> None:
    num_observed = 64
    values = torch.zeros(1024, 21, device=device)
    diagonal = torch.arange(20, device=device)
    values[diagonal, diagonal] = 1
    values[:, -1] = torch.arange(1024, device=device)
    table = TableTensor(
        columns={Stype.numerical: [*(f"x{i}" for i in range(20)), "target"]},
        numerical=values,
    )

    output = MaskNearEmpty(
        columns=(1.0, 1.0),
        observed=(num_observed / 1024, num_observed / 1024),
        target="target",
    ).fit_transform(
        table=table,
        generator=torch.Generator(device=device).manual_seed(0),
    )

    missing = output.numerical.isnan()
    assert missing[:, :-1].sum(dim=0).eq(1024 - num_observed).all()
    observed = ~missing[:, :-1]
    overlap = observed.mT.int() @ observed.int()
    overlap.fill_diagonal_(-1)
    assert overlap.amax(dim=-1).ge(num_observed - 2).all()
    assert all(
        output.numerical[:, i][observed[:, i]].unique().numel() == 2
        for i in range(20)
    )
    assert output.numerical[:, -1].equal(table.numerical[:, -1])
    assert output.numerical[~missing].equal(table.numerical[~missing])
