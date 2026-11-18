# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor


def normalize(matrix: Tensor, generator: torch.Generator | None) -> Tensor:
    """Add Gaussian noise with std ``1e-6`` and L2-normalize rows."""
    noise = torch.randn(
        size=matrix.size(),
        dtype=matrix.dtype,
        device=matrix.device,
        generator=generator,
    )
    matrix.add_(noise, alpha=1e-6)
    matrix /= matrix.norm(dim=-1, keepdim=True).clamp(
        min=torch.finfo(matrix.dtype).eps
    )
    return matrix
