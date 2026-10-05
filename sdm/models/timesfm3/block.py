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
from typing import Any

import torch
from torch import Tensor
from torch.nn import Identity, Linear, ReLU, RMSNorm, Sequential

from sdm.nn import Attention, RotaryEmbedding, SoftplusScale


class TimesFM3Attention(Attention):
    """Apply TimesFM-3 query and key transforms with SDM attention.

    Args:
        channels: Input and output dimensions of the attention layer.
        num_query_heads: Number of query heads.
        use_rope: Whether to rotate queries and keys by position.
        qk_norm: Whether to apply RMS normalization to queries and keys.
        bias: Whether projection layers have biases.
        device: Device on which to create parameters and buffers.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        channels: int,
        num_query_heads: int,
        use_rope: bool = True,
        qk_norm: bool = True,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        head_dim = channels // num_query_heads

        transforms: list[Sequential] = []
        for is_query in (True, False):
            rope = (
                RotaryEmbedding(
                    channels=head_dim,
                    layout="split_half",
                    theta=10_000,
                    requires_grad=False,
                    device=device,
                    dtype=torch.float32,
                )
                if use_rope
                else Identity()
            )

            norm = (
                RMSNorm(head_dim, **factory_kwargs) if qk_norm else Identity()
            )

            layers: OrderedDict[str, torch.nn.Module] = OrderedDict(
                rope=rope, norm=norm
            )

            if is_query:
                layers["scale"] = SoftplusScale(
                    channels=head_dim,
                    multiplier=1 / math.log(2),
                    **factory_kwargs,
                )
            transforms.append(Sequential(layers))

        super().__init__(
            channels=channels,
            num_query_heads=num_query_heads,
            query_transform=transforms[0],
            key_transform=transforms[1],
            scale=1.0,
            bias=bias,
            **factory_kwargs,
        )


class TimesFM3TransformerBlock(torch.nn.Module):
    """TimesFM-3 transformer block from Jain and Sen (2026).

    Args:
        channels: Input, output, and feed-forward width.
        num_heads: Number of heads in each attention layer.
        device: Device on which to create parameters and buffers.
        dtype: Data type of the parameters.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        # Time attention
        self.pre_time_attn = RMSNorm(channels, **factory_kwargs)
        self.post_time_attn = RMSNorm(channels, **factory_kwargs)
        self.time_attn = TimesFM3Attention(
            channels, num_heads, use_rope=True, **factory_kwargs
        )

        # Variate attention
        self.pre_var_attn = RMSNorm(channels, **factory_kwargs)
        self.post_var_attn = RMSNorm(channels, **factory_kwargs)
        self.var_attn = TimesFM3Attention(
            channels, num_heads, use_rope=False, **factory_kwargs
        )

        # Feed-forward block
        self.ff_block = Sequential(
            OrderedDict(
                pre_ff=RMSNorm(channels, **factory_kwargs),
                ff0=Linear(channels, channels, bias=False, **factory_kwargs),
                activation=ReLU(),
                ff1=Linear(channels, channels, bias=False, **factory_kwargs),
                post_ff=RMSNorm(channels, **factory_kwargs),
            )
        )

    def forward(
        self,
        x: Tensor,  # [B, V, N, C]
        patch_mask: Tensor,  # [B, V, N]
    ) -> Tensor:  # [B, V, N, C]
        """Apply time attention, variate attention, and an FFN.

        Args:
            x: Patch embeddings with shape ``[B, V, N, C]``.
            patch_mask: Boolean mask with shape ``[B, V, N]``. ``True``
                excludes a patch from attention keys.

        Returns:
            Updated embeddings with shape ``[B, V, N, C]``.
        """
        B, V, N, C = x.size()

        # Time attention
        time_input = self.pre_time_attn(x).reshape(B * V, N, C)
        time_attn_mask = self._make_time_attn_mask(patch_mask)

        time_output = self.time_attn(
            time_input, attn_mask=time_attn_mask
        ).reshape(B, V, N, C)
        hidden = x + self.post_time_attn(time_output)

        # Variate attention
        var_input = (
            self.pre_var_attn(hidden).permute(0, 2, 1, 3).reshape(B * N, V, C)
        )
        var_attn_mask = self._make_variate_attn_mask(patch_mask)

        var_output = self.var_attn(var_input, attn_mask=var_attn_mask)
        var_output = var_output.reshape(B, N, V, C).permute(0, 2, 1, 3)
        hidden = hidden + self.post_var_attn(var_output)

        ff_output = self.ff_block(hidden)
        return hidden + ff_output

    def _make_time_attn_mask(self, patch_mask: Tensor) -> Tensor:
        """Make a causal attention mask for time attention.

        Args:
            patch_mask: Boolean mask with shape ``[B, V, N]``. ``True``
                excludes a patch from attention keys.

        Returns:
            Boolean temporal attention mask with shape
            ``[B * V, N, N]``. ``True`` permits attention.
        """
        B, V, N = patch_mask.size()
        causal_mask = (
            torch.ones(N, N, device=patch_mask.device, dtype=torch.bool)
            .tril()
            .reshape(1, N, N)
        )
        return causal_mask & (~patch_mask).reshape(B * V, 1, N)

    def _make_variate_attn_mask(self, patch_mask: Tensor) -> Tensor:
        """Make a mask for variate attention.

        Args:
            patch_mask: Boolean mask with shape ``[B, V, N]``. ``True``
                excludes a patch from attention keys.

        Returns:
            Boolean variate attention mask with shape ``[B * N, 1, V]``.
            ``True`` permits attention.
        """
        B, V, N = patch_mask.size()
        return (~patch_mask).permute(0, 2, 1).reshape(B * N, 1, V)
