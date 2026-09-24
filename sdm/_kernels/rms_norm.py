# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.nn.rope import RotaryEmbedding

_RMSNorm: TypeAlias = Callable[
    [Tensor, Tensor | None, float, torch.dtype, tuple[Tensor, Tensor] | None],
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
    x: Tensor,  # [..., S, H, C] with `rope`, else [..., C]
    weight: Tensor | None,  # [C]
    eps: float,
    dtype: torch.dtype,
    rope: RotaryEmbedding | None = None,
) -> Tensor:  # [..., C]
    r"""Normalize the last dimension in single precision and return ``dtype``.

    Matches :func:`torch.nn.functional.rms_norm` on single-precision inputs
    followed by a cast to ``dtype``, without materializing either. With
    ``rope``, the input is rotated by it first.
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
        and (
            rope is None
            or (
                rope.layout == "split_half"
                and rope.rotary_channels == x.size(-1)
                and rope.inv_freq.dtype == torch.float32
                and rope.inv_freq.device == x.device
                and not (
                    torch.is_grad_enabled() and rope.inv_freq.requires_grad
                )
            )
        )
    ):
        cos_sin: tuple[Tensor, Tensor] | None = None
        if rope is not None:  # Rotation tables as in `RotaryEmbedding`:
            seq = torch.arange(
                x.size(-3), device=x.device, dtype=torch.float32
            )
            freq = seq.view(-1, 1) * rope.inv_freq.view(1, -1)  # [S, C // 2]
            cos_sin = (freq.cos().to(x.dtype), freq.sin().to(x.dtype))
        out = _triton_rms_norm(x, weight, eps, dtype, cos_sin)
        if out is not None:
            return out
    if rope is not None:
        x = rope(x)
    with torch.autocast(x.device.type, enabled=False):
        return F.rms_norm(
            x.float(),
            normalized_shape=(x.size(-1),),
            weight=weight.float() if weight is not None else None,
            eps=eps,
        ).to(dtype)
