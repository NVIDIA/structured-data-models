# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm._kernels import rmsnorm_cast


class RMSNorm(torch.nn.RMSNorm):
    """A :class:`torch.nn.RMSNorm` that dispatches to an efficient kernel."""

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        if (
            x.is_cuda
            and torch.is_autocast_enabled("cuda")
            and not torch.is_grad_enabled()
            and x.dtype in {torch.float16, torch.bfloat16, torch.float32}
            and x.shape[-1:] == self.normalized_shape
            and (
                self.weight is None
                or self.weight.dtype
                in {torch.float16, torch.bfloat16, torch.float32}
            )
        ):
            eps = (
                self.eps
                if self.eps is not None
                else torch.finfo(torch.float32).eps
            )
            return rmsnorm_cast(x, self.weight, eps)
        return super().forward(x)
