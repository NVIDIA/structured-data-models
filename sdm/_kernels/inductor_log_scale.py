# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile shared-length query scaling while preserving eager arithmetic."""

import torch
from torch import Tensor

# Explicit benchmark setting: multiple ranks/gate modes/inference contexts
# share this one code object. Fullgraph keeps any cache exhaustion observable.
# Dynamo's mutable setting is typed as its default literal.
torch._dynamo.config.recompile_limit = 64  # ty: ignore[invalid-assignment]


def native(
    query: Tensor,
    key_len: Tensor | int,
    head_scale: Tensor,
    gate: Tensor | None = None,
) -> Tensor:
    length = (
        key_len.float()
        if isinstance(key_len, Tensor)
        else query.new_full((), key_len, dtype=torch.float32)
    )
    log_key_len = length.clamp(min=1.0).log().to(query.dtype)
    head_scale = head_scale.to(query.dtype).reshape(
        *((1,) * (query.dim() - 2)), -1, 1
    )
    out = query * log_key_len[..., None, None] * head_scale
    if gate is not None:
        out = out * (1 + gate.tanh())
    return out


@torch.compile(
    backend="inductor",
    fullgraph=True,
    options={
        "compile_threads": 1,
        "emulate_precision_casts": True,
        "eager_numerics.use_pytorch_libdevice": True,
    },
)
def compiled(
    query: Tensor, key_len: int, head_scale: Tensor, gate: Tensor | None = None
) -> Tensor:
    # Match the handwritten kernel's HEADS/CHANNELS specialization only.
    # Request lengths and leading batch/sequence dimensions remain automatic.
    torch._dynamo.mark_static(query, -2)
    torch._dynamo.mark_static(query, -1)
    torch._dynamo.mark_static(head_scale, 0)
    if gate is not None:
        torch._dynamo.mark_static(gate, -2)
        torch._dynamo.mark_static(gate, -1)
    return native(query, key_len, head_scale, gate)


def log_scale(
    query: Tensor,
    key_len: Tensor | int,
    head_scale: Tensor,
    gate: Tensor | None = None,
) -> Tensor:
    inputs = (query, head_scale) if gate is None else (query, head_scale, gate)
    if (
        query.is_cuda
        and query.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and query.numel() > 0
        and isinstance(key_len, int)
        and -(2**63) <= key_len < 2**63
        and head_scale.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and all(x.device == query.device and x.is_contiguous() for x in inputs)
        and (
            gate is None
            or (gate.shape == query.shape and gate.dtype == query.dtype)
        )
        and not (
            torch.is_grad_enabled() and any(x.requires_grad for x in inputs)
        )
    ):
        return compiled(query, key_len, head_scale, gate)
    return native(query, key_len, head_scale, gate)
