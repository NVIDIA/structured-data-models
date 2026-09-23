# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm._kernels import rmsnorm_cast


class RMSNorm(torch.nn.RMSNorm):
    """Root mean square normalization with fused CUDA autocast inference.

    Inherits the parameters and checkpoint format of :class:`torch.nn.RMSNorm`.
    During CUDA autocast inference, one-dimensional normalization returns the
    configured autocast dtype. Other calls retain PyTorch's behavior.

    Args:
        normalized_shape: Input dimensions to normalize, starting at the last
            dimension.
        eps: Constant added to the mean square before normalization. If
            ``None``, uses PyTorch's default for the computation dtype.
        elementwise_affine: Whether to learn a scale for each normalized value.
        device: The parameter device.
        dtype: The parameter dtype.
    """

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
