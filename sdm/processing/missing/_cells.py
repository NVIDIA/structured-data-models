# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import CategoricalTensor, TableTensor


def mask_cells(table: TableTensor, missing: Tensor) -> TableTensor:
    r"""Mask numerical and categorical cells where ``missing`` is true."""
    num_numerical = table.numerical.size(-1)
    code = table.categorical.code.masked_fill(missing[..., num_numerical:], -1)
    return table.replace_blocks(
        numerical=table.numerical.masked_fill(
            missing[..., :num_numerical], torch.nan
        ),
        categorical=CategoricalTensor(
            code=code,
            categories=table.categorical.categories,
        ),
    )
