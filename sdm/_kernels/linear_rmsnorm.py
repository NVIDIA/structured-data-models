# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
from torch import Tensor

_implementation: Callable[..., Tensor] | None = None
if torch.version.cuda is not None:
    try:
        from sdm._kernels._linear_rmsnorm_cuda import run
    except ImportError:
        pass
    else:
        _implementation = run


def _linear_rmsnorm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    head_dim: int,
    eps: float,
    rope_inv_freq: Tensor | None = None,
    *,
    packed_key_value: bool = False,
) -> Tensor | None:
    if (
        _implementation is None
        or type(x) is not Tensor
        or not x.is_cuda
        or x.dtype != torch.float16
        or not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.float16
        or torch.is_grad_enabled()
        or not x.is_contiguous()
        or x.dim() < 2
        or head_dim not in {32, 64, 128}
        or x.size(-1) < head_dim
        or x.size(-1) % head_dim
        or x.size(-1) % 32
        # Wider reductions may use
        # native split-K, whose partial sums have a different rounding order.
        or x.size(-1) >= 1024
        or weight.shape
        != (x.size(-1) * (2 if packed_key_value else 1), x.size(-1))
        or (bias is not None and bias.shape != (weight.size(0),))
        or x.numel() < 32 * x.size(-1)
        or x.numel() // x.size(-1) >= 2**31 - 128
        or torch.backends.cuda.matmul.allow_fp16_accumulation
        or torch.backends.cuda.preferred_blas_library()
        != torch._C._BlasBackend.Cublas
        or torch.cuda.get_device_capability(x.device) != (12, 0)
    ):
        return None
    for tensor in (x, weight, bias):
        if tensor is not None and (
            type(tensor) not in {Tensor, torch.nn.Parameter}
            or tensor.dtype not in {torch.float16, torch.float32}
            or tensor.device != x.device
            or not tensor.is_contiguous()
            or tensor.data_ptr() % 16
            or tensor.is_neg()
            or tensor.is_conj()
        ):
            return None
    rope = None
    if rope_inv_freq is not None:
        if (
            type(rope_inv_freq) not in {Tensor, torch.nn.Parameter}
            or rope_inv_freq.device != x.device
            or rope_inv_freq.shape != (head_dim // 2,)
        ):
            return None
        seq = torch.arange(x.size(-2), device=x.device, dtype=torch.float32)
        freq = seq.view(-1, 1) * rope_inv_freq.view(1, -1)
        rope = (freq.cos().to(x.dtype), freq.sin().to(x.dtype))
    return _implementation(
        x.reshape(-1, x.size(-1)),
        weight,
        bias,
        head_dim,
        eps,
        rope[0] if rope is not None else None,
        rope[1] if rope is not None else None,
        packed_key_value,
    ).view(*x.shape[:-1], weight.size(0))
