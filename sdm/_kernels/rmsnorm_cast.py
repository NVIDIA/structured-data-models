# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

_RMSNormCast: TypeAlias = Callable[[Tensor, Tensor | None, float], Tensor]

_triton_rmsnorm_cast: _RMSNormCast | None = None
try:
    from sdm._kernels.triton.rmsnorm_cast import (
        rmsnorm_cast as _triton_rmsnorm_cast_impl,
    )
except ImportError:
    pass
else:
    _triton_rmsnorm_cast = _triton_rmsnorm_cast_impl


def rmsnorm_cast(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """Apply RMSNorm over the last dimension and cast to the autocast dtype.

    The output uses the configured autocast dtype for the input device,
    even when autocast is disabled.

    Args:
        x: Input tensor with shape ``[..., C]``, where ``C`` is the number
            of channels.
        weight: Scale tensor with shape ``[C]``, or ``None`` for no scaling.
        eps: Constant added to the mean square before normalization.
    """
    if (
        _triton_rmsnorm_cast is not None
        and x.is_cuda
        and x.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and x.size(-1) in {64, 128, 256, 512}
        and (
            weight is None
            or (
                weight.dtype in {torch.float16, torch.bfloat16, torch.float32}
                and weight.device == x.device
                and weight.shape == x.shape[-1:]
            )
        )
        and x.numel() > 0
        and not (
            (x.requires_grad or (weight is not None and weight.requires_grad))
            and torch.is_grad_enabled()
        )
    ):
        return _triton_rmsnorm_cast(x, weight, eps)
    return F.rms_norm(x, (x.size(-1),), weight, eps).to(
        torch.get_autocast_dtype(x.device.type)
    )
