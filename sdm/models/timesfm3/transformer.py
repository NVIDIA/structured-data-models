# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import Identity, RMSNorm, Sequential

from sdm.nn import Attention, RotaryEmbedding, SoftplusScale


def make_attn_mask(patch_mask: Tensor, causal: bool = True) -> Tensor:
    """Create an attention mask in which ``True`` permits attention.

    Args:
        patch_mask: Masked patches with shape ``[B, N]``.
        causal: Whether queries may attend only to preceding positions.

    Returns:
        Boolean mask with shape ``[B, 1, N, N]`` when causal and broadcastable
        shape ``[B, 1, 1, N]`` otherwise.
    """
    mask = ~patch_mask[:, None, None, :]
    if not causal:
        return mask
    causal_mask = torch.ones(
        patch_mask.size(1),
        patch_mask.size(1),
        dtype=torch.bool,
        device=patch_mask.device,
    ).tril()
    return causal_mask[None, None] & mask


class TimesFM3Attention(Attention):
    """Apply TimesFM-3 query and key transforms with SDM attention.

    Args:
        model_dims: Input and output width.
        num_heads: Number of attention heads.
        use_rope: Whether to rotate queries and keys by position.
        qk_norm: Query and key normalization.
        use_bias: Whether projection layers have biases.
        use_memory_efficient_attention: Whether to retain TimesFM's
            square-root head-dimension logit scaling.
        device: Device on which to create parameters and buffers.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        model_dims: int,
        num_heads: int,
        use_rope: bool = True,
        qk_norm: Literal["rms", "none"] = "rms",
        use_bias: bool = False,
        use_memory_efficient_attention: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        head_dim = model_dims // num_heads

        transforms: list[Sequential] = []
        for is_query in (True, False):
            rope: torch.nn.Module
            if use_rope:
                rope = RotaryEmbedding(
                    channels=head_dim,
                    layout="split_half",
                    theta=10_000,
                    requires_grad=False,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                rope = Identity()
            if qk_norm == "rms":
                norm: torch.nn.Module = RMSNorm(head_dim, **factory_kwargs)
            else:
                assert qk_norm == "none"
                norm = Identity()
            layers = OrderedDict(rope=rope, norm=norm)
            if is_query:
                layers["scale"] = SoftplusScale(
                    channels=head_dim,
                    multiplier=1.442695041 / math.sqrt(head_dim),
                    **factory_kwargs,
                )
            transforms.append(Sequential(layers))

        super().__init__(
            channels=model_dims,
            num_query_heads=num_heads,
            query_transform=transforms[0],
            key_transform=transforms[1],
            scale=(
                math.sqrt(head_dim) if use_memory_efficient_attention else 1.0
            ),
            bias=use_bias,
            **factory_kwargs,
        )
