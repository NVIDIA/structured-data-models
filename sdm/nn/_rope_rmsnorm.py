# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor
from torch.nn import RMSNorm, Sequential

from sdm.nn.rope import RotaryEmbedding

_rope_rmsnorm: Callable[..., Tensor] | None = None
try:
    from sdm._kernels.triton.rope_rmsnorm import rope_rmsnorm
except ImportError:
    pass
else:
    _rope_rmsnorm = rope_rmsnorm


class _RoPERMSNorm(Sequential):
    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        x = input
        # Keep the original operations visible to whole-model compilation.
        if torch.compiler.is_compiling():
            return super().forward(x)

        rope = cast(RotaryEmbedding, self[0])
        norm = cast(RMSNorm, self[1])
        if (
            _rope_rmsnorm is None
            or self.training
            or torch.is_grad_enabled()
            or not x.is_cuda
            or torch.version.hip is not None
            or x.dtype not in {torch.float16, torch.bfloat16, torch.float32}
            or x.ndim < 3
            or x.size(-1) not in {32, 64}
            or x.size(-1) != rope.channels
            or x.numel() == 0
            or rope.inv_freq.dtype != torch.float32
            or rope.inv_freq.device != x.device
            or not rope.inv_freq.is_contiguous()
            or norm.weight is not None
        ):
            return super().forward(x)
        try:
            flat = x.view(-1, *x.shape[-3:])
        except RuntimeError:
            return super().forward(x)
        return _rope_rmsnorm(
            flat,
            rope.inv_freq,
            rotary_channels=rope.rotary_channels,
            interleaved=rope.layout == "interleaved",
            eps=cast(float, norm.eps),
        ).view(x.shape)
