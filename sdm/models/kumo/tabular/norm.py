# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import Tensor

_COMPILE_MIN_NUMEL = 5_000_000


@torch.compile(fullgraph=True)
def _compiled_rms_norm(
    x: Tensor,
    normalized_shape: tuple[int, ...],
    weight: Tensor | None,
    eps: float | None,
) -> Tensor:
    return F.rms_norm(x, normalized_shape, weight=weight, eps=eps)


class RMSNorm(torch.nn.RMSNorm):
    """RMS normalization compiled for large CUDA inputs."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization."""
        if (
            not x.is_cuda
            or x.numel() < _COMPILE_MIN_NUMEL
            or torch.compiler.is_compiling()
        ):
            return super().forward(x)

        torch._dynamo.mark_dynamic(x, tuple(range(x.ndim - 1)))
        return _compiled_rms_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.eps,
        )
