# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

_RMSNorm: TypeAlias = Callable[
    [Tensor, Tensor | None, float, torch.dtype],
    Tensor | None,
]

_triton_rms_norm: _RMSNorm | None = None
try:
    from sdm._kernels.triton.rms_norm import (
        rms_norm as _triton_rms_norm_impl,
    )
except ImportError:
    pass
else:
    _triton_rms_norm = _triton_rms_norm_impl


def rms_norm(
    x: Tensor,  # [..., C]
    weight: Tensor | None,  # [C]
    eps: float,
    dtype: torch.dtype,
) -> Tensor:  # [..., C]
    r"""Normalize the last dimension in single precision and return ``dtype``.

    Matches :func:`torch.nn.functional.rms_norm` on single-precision inputs
    followed by a cast to ``dtype``, without materializing either.
    """
    if (
        _triton_rms_norm is not None
        and not (
            torch.is_grad_enabled()
            and (
                x.requires_grad
                or (weight is not None and weight.requires_grad)
            )
        )
        and x.is_cuda
        and x.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and (
            weight is None
            or (
                weight.dtype == torch.float32
                and weight.device == x.device
                and weight.is_contiguous()
            )
        )
    ):
        out = _triton_rms_norm(x, weight, eps, dtype)
        if out is not None:
            return out
    with torch.autocast(x.device.type, enabled=False):
        return F.rms_norm(
            x.float(),
            normalized_shape=(x.size(-1),),
            weight=weight.float() if weight is not None else None,
            eps=eps,
        ).to(dtype)
