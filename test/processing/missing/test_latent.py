# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.processing import MaskLatent


def test_mask_latent_shares_a_hidden_row_pattern_across_columns() -> None:
    rows = torch.arange(200, dtype=torch.float32)[:, None]
    table = TableTensor.from_tensor(rows + torch.arange(8)[None, :])

    output = MaskLatent(
        num_proxies=(1, 1),
        columns=(1.0, 1.0),
        prevalence=(0.5, 0.5),
        rate_off=(0.0, 0.0),
        rate_on=(1.0, 1.0),
    ).fit_transform(table, generator=torch.Generator().manual_seed(0))

    missing = output.numerical.isnan()
    driven = missing.any(dim=0)
    assert driven.sum() == 7
    assert missing[:, driven].eq(missing[:, driven][:, :1]).all()
    assert missing[:, driven][:, 0].sum() == 100
    assert output.numerical[~missing].equal(table.numerical[~missing])
