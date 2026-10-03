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
from torch.nn import Identity, Linear, ReLU, RMSNorm, Sequential

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
            layers: OrderedDict[str, torch.nn.Module] = OrderedDict(
                rope=rope, norm=norm
            )
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


class MixingTransformer(torch.nn.Module):
    """Apply TimesFM-3 temporal, variate, and feed-forward mixing.

    Args:
        model_dims: Input and output width.
        hidden_dims: Feed-forward hidden width.
        num_heads: Number of attention heads.
        qk_norm: Query and key normalization.
        use_bias: Whether linear layers have biases.
        use_rope_seq: Whether temporal attention uses rotary embeddings.
        use_rope_var: Whether variate attention uses rotary embeddings.
        use_variate_attention: Whether to attend across variates.
        causal_attention: Whether temporal attention is causal.
        use_memory_efficient_attention: Whether to retain TimesFM's
            square-root head-dimension logit scaling.
        device: Device on which to create parameters and buffers.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        model_dims: int,
        hidden_dims: int,
        num_heads: int,
        qk_norm: Literal["rms", "none"],
        use_bias: bool,
        use_rope_seq: bool,
        use_rope_var: bool,
        use_variate_attention: bool = True,
        causal_attention: bool = True,
        use_memory_efficient_attention: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.use_variate_attention = use_variate_attention
        self.causal_attention = causal_attention

        self.pre_seq_attn_ln = RMSNorm(model_dims, **factory_kwargs)
        self.post_seq_attn_ln = RMSNorm(model_dims, **factory_kwargs)
        self.seq_attn = TimesFM3Attention(
            model_dims=model_dims,
            num_heads=num_heads,
            use_rope=use_rope_seq,
            qk_norm=qk_norm,
            use_bias=use_bias,
            use_memory_efficient_attention=use_memory_efficient_attention,
            **factory_kwargs,
        )

        if use_variate_attention:
            self.pre_var_attn_ln = RMSNorm(model_dims, **factory_kwargs)
            self.post_var_attn_ln = RMSNorm(model_dims, **factory_kwargs)
            self.var_attn = TimesFM3Attention(
                model_dims=model_dims,
                num_heads=num_heads,
                use_rope=use_rope_var,
                qk_norm=qk_norm,
                use_bias=use_bias,
                use_memory_efficient_attention=use_memory_efficient_attention,
                **factory_kwargs,
            )

        self.pre_ff_ln = RMSNorm(model_dims, **factory_kwargs)
        self.post_ff_ln = RMSNorm(model_dims, **factory_kwargs)
        self.ff0 = Linear(
            model_dims, hidden_dims, bias=use_bias, **factory_kwargs
        )
        self.ff1 = Linear(
            hidden_dims, model_dims, bias=use_bias, **factory_kwargs
        )
        self.activation = ReLU()

    def forward(
        self,
        input_embeddings: Tensor,
        patch_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Mix embeddings with shape ``[B, V, N, D]``.

        Args:
            input_embeddings: Patch embeddings with shape
                ``[B, V, N, D]``.
            patch_mask: Masked patches with shape ``[B, V, N]``.

        Returns:
            Mixed embeddings and the temporal attention mask with shape
            ``[B * V, 1, N, N]`` when causal and
            ``[B * V, 1, 1, N]`` otherwise.
        """
        batch_size, num_variates, num_patches, model_dims = (
            input_embeddings.shape
        )
        seq_input = self.pre_seq_attn_ln(input_embeddings).reshape(
            batch_size * num_variates, num_patches, model_dims
        )
        seq_patch_mask = patch_mask.reshape(
            batch_size * num_variates, num_patches
        )
        seq_attn_mask = make_attn_mask(
            seq_patch_mask, causal=self.causal_attention
        )
        seq_output = self.seq_attn(
            seq_input, attn_mask=seq_attn_mask.squeeze(1)
        )
        seq_output = seq_output.reshape(
            batch_size, num_variates, num_patches, model_dims
        )
        hidden = self.post_seq_attn_ln(seq_output) + input_embeddings

        if self.use_variate_attention:
            var_input = self.pre_var_attn_ln(hidden)
            var_input = var_input.permute(0, 2, 1, 3).reshape(
                batch_size * num_patches, num_variates, model_dims
            )
            var_patch_mask = patch_mask.permute(0, 2, 1).reshape(
                batch_size * num_patches, num_variates
            )
            var_output = self.var_attn(
                var_input, attn_mask=~var_patch_mask[:, None, :]
            )
            var_output = var_output.reshape(
                batch_size, num_patches, num_variates, model_dims
            ).permute(0, 2, 1, 3)
            hidden = self.post_var_attn_ln(var_output) + hidden

        ff_output = self.ff1(self.activation(self.ff0(self.pre_ff_ln(hidden))))
        return self.post_ff_ln(ff_output) + hidden, seq_attn_mask
