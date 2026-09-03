# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor


def _constant_feature_mask(
    var: Tensor,
    mean: Tensor,
    n_samples: int,
) -> Tensor:
    eps = torch.finfo(var.dtype).eps
    upper_bound = n_samples * eps * var + (n_samples * mean * eps) ** 2
    return var <= upper_bound
