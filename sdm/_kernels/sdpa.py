# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.fx.experimental.symbolic_shapes import statically_known_true

from sdm._memory import _split_tensor


def _sdpa(
    query: Tensor,  # [B, Hq, Q, C]
    key: Tensor,  # [B, Hkv, KV, C]
    value: Tensor,  # [B, Hkv, KV, C]
    attn_mask: Tensor | None,  # [B, Hq, Q, KV]
    dropout_p: float,
    enable_gqa: bool,
    is_causal: bool,
    scale: float | None,
) -> Tensor:  # [B, Q, Hq, C]
    # CUDA attention kernels accept at most 65,535 batch entries per call.
    queries = _split_tensor(query, 65_535, dim=0)
    masks = (
        (None,) * len(queries)
        if attn_mask is None
        else _split_tensor(attn_mask, 65_535, dim=0)
    )
    outs = []
    for chunk_query, chunk_key, chunk_value, chunk_mask in zip(
        queries,
        _split_tensor(key, 65_535, dim=0),
        _split_tensor(value, 65_535, dim=0),
        masks,
        strict=True,
    ):
        out = F.scaled_dot_product_attention(
            query=chunk_query,
            key=chunk_key,
            value=chunk_value,
            attn_mask=chunk_mask,
            dropout_p=dropout_p,
            enable_gqa=enable_gqa,
            is_causal=is_causal,
            scale=scale,
        )
        outs.append(out.transpose(-3, -2))
    return outs[0] if len(outs) == 1 else torch.cat(outs)


@torch.library.custom_op("sdm::sdpa", mutates_args=())
def _sdpa_operator(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None,
    dropout_p: float,
    enable_gqa: bool,
    is_causal: bool,
    scale: float | None,
) -> Tensor:
    """Run `_sdpa` as one operator that `torch.compile` does not trace.

    Tracing its loop would recompile whenever the number of slices changes.
    """
    # Compiled code expects a contiguous output, as the fake below declares.
    return _sdpa(
        query, key, value, attn_mask, dropout_p, enable_gqa, is_causal, scale
    ).contiguous()


@_sdpa_operator.register_fake
def _sdpa_operator_fake(
    query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs: Any
) -> Tensor:
    return query.new_empty(
        (query.size(0), query.size(2), query.size(1), value.size(-1))
    )


def sdpa(
    query: Tensor,  # [B, Hq, Q, C]
    key: Tensor,  # [B, Hkv, KV, C]
    value: Tensor,  # [B, Hkv, KV, C]
    attn_mask: Tensor | None,  # [B, Hq, Q, KV]
    dropout_p: float,
    enable_gqa: bool,
    is_causal: bool,
    scale: float | None,
) -> Tensor:  # [B, Q, Hq, C]
    # The operator avoids a recompile when the number of slices changes.
    # It has no gradients, and a batch known to need no slicing skips it.
    kernel = _sdpa
    if not torch.is_grad_enabled() and not statically_known_true(
        query.size(0) <= 65_535
    ):
        kernel = _sdpa_operator
        device_type = query.device.type
        if torch.is_autocast_enabled(device_type):
            # Compiled code skips autocast inside the operator, so
            # convert the inputs here. Autocast never converts float64.
            dtype = torch.get_autocast_dtype(device_type)
            query, key, value = (
                x if x.dtype == torch.float64 else x.to(dtype)
                for x in (query, key, value)
            )
    return kernel(
        query, key, value, attn_mask, dropout_p, enable_gqa, is_causal, scale
    )
