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
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.models.timesfm3.normalization import PerDimScale


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


class RotaryPositionalEmbedding(torch.nn.Module):
    """Apply RoPE from the `RoFormer paper <https://arxiv.org/abs/2104.09864>`_.

    Args:
        embedding_dims: Size of the last input dimension.
        device: Device on which to create the timescale buffer.
    """

    timescale: Tensor

    def __init__(
        self,
        embedding_dims: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dims = embedding_dims
        self.register_buffer(
            "timescale",
            self._make_timescale(device),
            persistent=False,
        )

    def _make_timescale(self, device: torch.device | str | None) -> Tensor:
        half_dim = self.embedding_dims // 2
        fraction = (
            2.0
            * torch.arange(half_dim, dtype=torch.float32, device=device)
            / self.embedding_dims
        )
        return 10_000**fraction

    def _reset_timescale(self, device: torch.device | str) -> None:
        self.timescale = self._make_timescale(device)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply rotary positional embeddings.

        Args:
            inputs: Input with shape ``[B, S, D]`` or ``[B, S, H, D]``.

        Returns:
            Rotated input with the same shape as ``inputs``.
        """
        if self.embedding_dims != inputs.shape[-1]:
            raise ValueError(
                "The embedding dims of the rotary position embedding must "
                "match the hidden dimension of the inputs."
            )

        position = torch.arange(
            inputs.shape[1],
            device=inputs.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        if inputs.dim() == 4:
            position = position.unsqueeze(-1).unsqueeze(-1)
            timescale = self.timescale.view(1, 1, 1, -1)
        elif inputs.dim() == 3:
            position = position.unsqueeze(-1)
            timescale = self.timescale.view(1, 1, -1)
        else:
            raise ValueError("Inputs must be of rank 3 or 4.")

        sinusoid = position.float() / timescale
        sin = sinusoid.sin().to(inputs.dtype)
        cos = sinusoid.cos().to(inputs.dtype)
        first_half, second_half = inputs.chunk(2, dim=-1)
        first = first_half * cos - second_half * sin
        second = second_half * cos + first_half * sin
        return torch.cat((first, second), dim=-1)


class MultiHeadAttention(torch.nn.Module):
    """Apply TimesFM-3 multi-head attention.

    Args:
        num_heads: Number of attention heads.
        in_features: Input and output dimension.
        use_rotary_position_embeddings: Whether to apply rotary positional
            embeddings to queries and keys.
        causal_attention: Whether queries can only attend to preceding keys.
        use_bias: Whether projection layers use bias parameters.
        qk_norm: Query and key normalization.
        v_norm: Value normalization.
        use_sdpa: Whether to use PyTorch scaled dot-product attention.
        rescale_logits: Whether to cancel TimesFM's square-root
            head-dimension logit scaling.
        device: Device on which to create parameters and buffers.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        use_rotary_position_embeddings: bool = True,
        causal_attention: bool = True,
        use_bias: bool = False,
        qk_norm: Literal["rms", "none"] = "rms",
        v_norm: Literal["rms", "none"] = "none",
        use_sdpa: bool = False,
        rescale_logits: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_heads = num_heads
        self.in_features = in_features
        self.causal_attention = causal_attention
        self.head_dim = in_features // num_heads
        self.use_sdpa = use_sdpa
        self.rescale_logits = rescale_logits

        self.query_proj = torch.nn.Linear(
            in_features,
            in_features,
            bias=use_bias,
            **factory_kwargs,
        )
        self.key_proj = torch.nn.Linear(
            in_features,
            in_features,
            bias=use_bias,
            **factory_kwargs,
        )
        self.value_proj = torch.nn.Linear(
            in_features,
            in_features,
            bias=use_bias,
            **factory_kwargs,
        )
        self.out_proj = torch.nn.Linear(
            in_features,
            in_features,
            bias=use_bias,
            **factory_kwargs,
        )

        if qk_norm == "rms":
            self.query_ln = torch.nn.RMSNorm(
                self.head_dim,
                **factory_kwargs,
            )
            self.key_ln = torch.nn.RMSNorm(
                self.head_dim,
                **factory_kwargs,
            )
        elif qk_norm == "none":
            self.query_ln = None
            self.key_ln = None
        else:
            raise AssertionError(f"Unhandled QK normalization: {qk_norm}")

        if v_norm == "rms":
            self.value_ln = torch.nn.RMSNorm(
                self.head_dim,
                elementwise_affine=False,
                **factory_kwargs,
            )
        elif v_norm == "none":
            self.value_ln = None
        else:
            raise AssertionError(f"Unhandled value normalization: {v_norm}")

        if use_rotary_position_embeddings:
            self.rotary_position_embedding = RotaryPositionalEmbedding(
                embedding_dims=self.head_dim,
                device=device,
            )
        else:
            self.rotary_position_embedding = None

        self.per_dim_scale = PerDimScale(
            num_dims=self.head_dim,
            **factory_kwargs,
        )

        self.register_load_state_dict_post_hook(self._materialize_rope)

    def _materialize_rope(
        self,
        module: torch.nn.Module,
        incompatible_keys: Any,
    ) -> None:
        del module, incompatible_keys
        if self.rotary_position_embedding is not None:
            self.rotary_position_embedding._reset_timescale(
                self.query_proj.weight.device
            )

    def forward(
        self,
        inputs_q: Tensor,
        *,
        patch_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply multi-head attention.

        Args:
            inputs_q: Inputs with shape ``[B, N, D]``.
            patch_mask: Masked patches with shape ``[B, N]``.

        Returns:
            Attention output with shape ``[B, N, D]`` and the attention mask.
        """
        batch_size, num_patches, _ = inputs_q.shape
        if patch_mask is None:
            patch_mask = torch.zeros(
                batch_size,
                num_patches,
                dtype=torch.bool,
                device=inputs_q.device,
            )

        projection_shape = (
            batch_size,
            num_patches,
            self.num_heads,
            self.head_dim,
        )
        query = self.query_proj(inputs_q).view(projection_shape)
        key = self.key_proj(inputs_q).view(projection_shape)
        value = self.value_proj(inputs_q).view(projection_shape)

        if self.rotary_position_embedding is not None:
            query = self.rotary_position_embedding(query)
            key = self.rotary_position_embedding(key)

        if self.query_ln is not None:
            query = self.query_ln(query)
        if self.key_ln is not None:
            key = self.key_ln(key)
        query = self.per_dim_scale(query)
        if self.value_ln is not None:
            value = self.value_ln(value)

        attn_mask = make_attn_mask(patch_mask, causal=self.causal_attention)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        expanded_mask = attn_mask.expand(-1, self.num_heads, -1, -1)
        if self.use_sdpa:
            scale = 1.0 if self.rescale_logits else math.sqrt(self.head_dim)
            x = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=expanded_mask,
                scale=scale,
            )
        else:
            query = query * math.sqrt(self.head_dim)
            logits = query.matmul(key.transpose(-2, -1))
            if self.rescale_logits:
                logits = logits / math.sqrt(self.head_dim)
            mask_value = max(-1e9, torch.finfo(logits.dtype).min)
            mask_bias = torch.zeros_like(logits).masked_fill(
                ~expanded_mask,
                mask_value,
            )
            logits = logits + mask_bias
            x = logits.softmax(dim=-1).matmul(value)

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_patches, self.in_features)
        )
        return self.out_proj(x), attn_mask
