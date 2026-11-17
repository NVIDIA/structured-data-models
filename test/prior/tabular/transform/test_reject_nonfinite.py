# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.prior.tabular.transform import RejectNonfinite


def test_reject_nonfinite() -> None:
    x = TableTensor.from_tensor(torch.ones(4, 2))
    y = TableTensor.from_tensor(torch.ones(4, 1))
    transform = RejectNonfinite()

    result = transform(x, y)

    assert result is not None
    assert result[0] is x
    assert result[1] is y

    x.numerical[0, 0] = torch.nan
    assert transform(x, y) is None

    x.numerical[0, 0] = 1.0
    y.numerical[0, 0] = torch.inf
    assert transform(x, y) is None
