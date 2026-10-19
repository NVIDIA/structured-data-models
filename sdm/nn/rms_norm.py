# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from torch.nn.modules import module as module_hooks

from sdm._kernels import rmsnorm_cast
from sdm._kernels.rmsnorm_cast import _rmsnorm_pair
from sdm.nn.rope import RotaryEmbedding


class RMSNorm(torch.nn.RMSNorm):
    """A :class:`torch.nn.RMSNorm` that dispatches to an efficient kernel."""

    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding | None = None,
    ) -> Tensor:
        """Apply optional rotary embeddings before normalization.

        Args:
            x: Input with shape ``[..., C]``, or ``[..., S, H, C]`` with
                rotary embeddings, where ``S`` is the sequence length,
                ``H`` is the number of heads, and ``C`` is channels per head.
            rope: Optional rotary embedding applied to the input.
        """
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
            cos_sin: tuple[Tensor, Tensor] | None = None
            if rope is not None:
                if (
                    not torch.compiler.is_compiling()
                    and rope.layout == "split_half"
                    and rope.rotary_channels == rope.channels == x.size(-1)
                ):
                    seq = torch.arange(
                        x.size(-3), device=x.device, dtype=torch.float32
                    )
                    freq = seq.view(-1, 1) * rope.inv_freq.view(1, -1)
                    cos_sin = (freq.cos().to(x.dtype), freq.sin().to(x.dtype))
                else:
                    x = rope(x)
            eps = (
                self.eps
                if self.eps is not None
                else torch.finfo(torch.float32).eps
            )
            return rmsnorm_cast(x, self.weight, eps, rope=cos_sin)
        return super().forward(x if rope is None else rope(x))


def _shared_normalization(
    first: torch.nn.Module | None,
    second: torch.nn.Module | None,
    x: Tensor,
) -> tuple[Tensor, Tensor] | None:
    if (
        torch.compiler.is_compiling()
        or torch.jit.is_tracing()
        or torch.is_grad_enabled()
        or not torch.is_autocast_enabled("cuda")
        or type(first) is not RMSNorm
        or type(second) is not RMSNorm
        or first is second
        or first.normalized_shape != x.shape[-1:]
        or second.normalized_shape != x.shape[-1:]
    ):
        return None
    # Module hooks and customized calls retain the ordinary module path.
    if (
        module_hooks._global_forward_hooks
        or module_hooks._global_forward_pre_hooks
        or module_hooks._global_backward_hooks
        or module_hooks._global_backward_pre_hooks
        or any(
            norm._forward_hooks
            or norm._forward_pre_hooks
            or norm._backward_hooks
            or norm._backward_pre_hooks
            or norm._compiled_call_impl is not None
            or "forward" in norm.__dict__
            for norm in (first, second)
        )
    ):
        return None
    return _rmsnorm_pair(
        x,
        first.weight,
        second.weight,
        first.eps if first.eps is not None else torch.finfo(torch.float32).eps,
        second.eps
        if second.eps is not None
        else torch.finfo(torch.float32).eps,
    )
