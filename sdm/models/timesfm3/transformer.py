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
from torch.nn import Linear, ReLU, RMSNorm, Sequential

from sdm.nn import Attention, RotaryEmbedding, SoftplusScale


def _attention(
    channels: int,
    num_heads: int,
    use_rope: bool,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> Attention:
    head_dim = channels // num_heads
    factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
    transforms = []
    for is_query in (True, False):
        layers: OrderedDict[str, torch.nn.Module] = OrderedDict()
        if use_rope:
            layers["rope"] = RotaryEmbedding(
                channels=head_dim,
                layout="split_half",
                theta=10_000,
                requires_grad=False,
                device=device,
                dtype=torch.float32,
            )
        layers["norm"] = RMSNorm(head_dim, **factory_kwargs)
        if is_query:
            layers["scale"] = SoftplusScale(
                channels=head_dim,
                multiplier=1 / math.log(2),
                **factory_kwargs,
            )
        transforms.append(Sequential(layers))

    return Attention(
        channels=channels,
        num_query_heads=num_heads,
        query_transform=transforms[0],
        key_transform=transforms[1],
        scale=1.0,
        bias=False,
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

        self.pre_seq_attn_ln = RMSNorm(channels, **factory_kwargs)
        self.post_seq_attn_ln = RMSNorm(channels, **factory_kwargs)
        self.seq_attn = _attention(channels, num_heads, True, device, dtype)

        self.pre_var_attn_ln = RMSNorm(channels, **factory_kwargs)
        self.post_var_attn_ln = RMSNorm(channels, **factory_kwargs)
        self.var_attn = _attention(channels, num_heads, False, device, dtype)

        self.pre_ff_ln = RMSNorm(channels, **factory_kwargs)
        self.post_ff_ln = RMSNorm(channels, **factory_kwargs)
        self.ff0 = Linear(channels, channels, bias=False, **factory_kwargs)
        self.ff1 = Linear(channels, channels, bias=False, **factory_kwargs)
        self.activation = ReLU()

    def forward(
        self,
        x: Tensor,  # [B, V, N, C]
        patch_mask: Tensor,  # [B, V, N]
    ) -> tuple[Tensor, Tensor]:  # [B, V, N, C], [B * V, 1, N, N]
        """Apply time attention, variate attention, and an FFN.

        Args:
            x: Patch embeddings with shape ``[B, V, N, C]``.
            patch_mask: Masked patches with shape ``[B, V, N]``. ``True``
                indicates a masked patch.

        Returns:
            Updated embeddings with shape ``[B, V, N, C]`` and the temporal
            attention mask with shape ``[B * V, 1, N, N]``. ``True`` permits
            attention.
        """
        batch_size, num_variates, num_patches, channels = x.shape
        seq_input = self.pre_seq_attn_ln(x).reshape(
            batch_size * num_variates, num_patches, channels
        )
        seq_patch_mask = patch_mask.reshape(
            batch_size * num_variates, num_patches
        )
        causal_mask = torch.ones(
            num_patches, num_patches, device=x.device, dtype=torch.bool
        ).tril()
        seq_attn_mask = (
            causal_mask[None, None] & ~seq_patch_mask[:, None, None, :]
        )
        seq_output = self.seq_attn(
            seq_input, attn_mask=seq_attn_mask.squeeze(1)
        ).reshape(batch_size, num_variates, num_patches, channels)
        hidden = x + self.post_seq_attn_ln(seq_output)

        var_input = (
            self.pre_var_attn_ln(hidden)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * num_patches, num_variates, channels)
        )
        var_patch_mask = patch_mask.permute(0, 2, 1).reshape(
            batch_size * num_patches, num_variates
        )
        var_output = self.var_attn(
            var_input, attn_mask=~var_patch_mask[:, None, :]
        )
        var_output = var_output.reshape(
            batch_size, num_patches, num_variates, channels
        ).permute(0, 2, 1, 3)
        hidden = hidden + self.post_var_attn_ln(var_output)

        ff_output = self.ff1(self.activation(self.ff0(self.pre_ff_ln(hidden))))
        return hidden + self.post_ff_ln(ff_output), seq_attn_mask
