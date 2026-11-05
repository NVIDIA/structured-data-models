# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
from torch import Tensor
from torch._higher_order_ops import map as torch_map


def chunked_map(
    function: Callable[[Tensor], Tensor],
    x: Tensor,
    chunk_size: int,
    dim: int = 0,
) -> Tensor:
    """Map a pure, row-independent function preserving the selected dimension.

    Inputs fitting in one chunk are processed directly. Otherwise, the last
    chunk repeats the last input row as padding; repeated output rows are
    discarded. The function must not use randomness, mutate state, or mix rows.
    Output layout is unspecified and its storage can include padded rows.
    This uses a private PyTorch operator and does not support training. Eager
    callers should retain their existing loop rather than using this helper.
    """
    dim = dim if dim >= 0 else x.ndim + dim
    rows = x.size(dim)
    if rows <= chunk_size:
        return function(x)

    offsets = torch.arange(chunk_size, device=x.device)

    def apply_chunk(chunk: Tensor) -> Tensor:
        indices = (chunk * chunk_size + offsets).clamp_max(rows - 1)
        result = function(x.index_select(dim, indices))
        return result.movedim(dim, 0).contiguous()

    chunks = torch.arange(
        (rows + chunk_size - 1) // chunk_size, device=x.device
    )
    # PyTorch types this callback with an extra argument it does not pass.
    result = torch_map(apply_chunk, chunks)  # ty: ignore[invalid-argument-type]
    return result.flatten(0, 1).narrow(0, 0, rows).movedim(0, dim)
