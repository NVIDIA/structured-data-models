# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.prior import Uniform
from sdm.prior.tabular.transform import MaskLatent


def test_mask_latent_shares_a_hidden_row_pattern_across_columns() -> None:
    rows = torch.arange(200, dtype=torch.float32)[:, None]
    x = TableTensor.from_tensor(rows + torch.arange(8)[None, :])
    y = TableTensor.from_tensor(rows)

    output, output_y = MaskLatent(
        num_proxies=(1, 1),
        columns=(1.0, 1.0),
        prevalence=0.5,
        rate_off=Uniform(0.0, 0.0),
        rate_on=Uniform(1.0, 1.0),
    )(x=x, y=y)

    missing = output.numerical.isnan()
    driven = missing.any(dim=0)
    assert driven.sum() == 7
    assert missing[:, driven].eq(missing[:, driven][:, :1]).all()
    assert missing[:, driven][:, 0].sum() == 100
    assert output.numerical[~missing].equal(x.numerical[~missing])
    assert output_y.numerical.equal(y.numerical)
