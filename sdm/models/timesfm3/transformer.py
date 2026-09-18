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
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.models.timesfm3.configs import (
    StackedTransformersConfig,
    TransformerConfig,
)
from sdm.models.timesfm3.normalization import PerDimScale
from sdm.models.timesfm3.util import DecodeCache, get_activation_fn


def make_attn_mask(
    query_length: int,
    num_all_masked_kv: Tensor,
    query_index_offset: Tensor | None = None,
    kv_length: int = 0,
    causal: bool = True,
) -> Tensor:
    """Create an attention mask in which ``True`` permits attention.

    Args:
        query_length: Number of query positions ``Q``.
        num_all_masked_kv: Number of leading masked key/value positions with
            shape ``[B]``.
        query_index_offset: Optional query-position offsets with shape ``[B]``
            for cached decoding.
        kv_length: Number of key/value positions ``K``. A value of zero uses
            ``query_length``.
        causal: Whether queries may attend only to preceding positions.

    Returns:
        Boolean mask with shape ``[B, 1, Q, K]`` when causal and broadcastable
        shape ``[B, 1, 1, K]`` otherwise.
    """
    if kv_length == 0:
        kv_length = query_length

    query_index = torch.arange(
        query_length,
        device=num_all_masked_kv.device,
    ).view(1, 1, -1, 1)
    if query_index_offset is not None:
        query_index = query_index + query_index_offset.view(-1, 1, 1, 1)
    kv_index = torch.arange(
        kv_length,
        device=num_all_masked_kv.device,
    ).view(1, 1, 1, -1)
    mask = kv_index >= num_all_masked_kv.view(-1, 1, 1, 1)
    if causal:
        return (query_index >= kv_index) & mask
    return mask


def make_segment_mask(segment_ids: Tensor) -> Tensor:
    """Create an attention mask that isolates packed segments.

    Args:
        segment_ids: Segment identifiers with shape ``[B, S]``.

    Returns:
        Boolean mask with shape ``[B, 1, S, S]``.
    """
    return (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)


class RotaryPositionalEmbedding(torch.nn.Module):
    """Apply RoPE from the `RoFormer paper <https://arxiv.org/abs/2104.09864>`_.

    Args:
        embedding_dims: Size of the last input dimension.
        min_timescale: Minimum rotation timescale.
        max_timescale: Maximum rotation timescale.
        device: Device on which to create the timescale buffer.
    """

    timescale: Tensor

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10_000,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

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
        return (
            self.min_timescale
            * (self.max_timescale / self.min_timescale) ** fraction
        )

    def _reset_timescale(self, device: torch.device | str) -> None:
        self.timescale = self._make_timescale(device)

    def forward(
        self,
        inputs: Tensor,
        position: Tensor | None = None,
    ) -> Tensor:
        """Apply rotary positional embeddings.

        Args:
            inputs: Input with shape ``[B, S, D]`` or ``[B, S, H, D]``.
            position: Optional positions with shape ``[B, S]``. Sequential
                positions are used by default.

        Returns:
            Rotated input with the same shape as ``inputs``.
        """
        if self.embedding_dims != inputs.shape[-1]:
            raise ValueError(
                "The embedding dims of the rotary position embedding must "
                "match the hidden dimension of the inputs."
            )

        if position is None:
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
        use_per_dim_scale: Whether to scale each query-head dimension.
        use_rotary_position_embeddings: Whether to apply rotary positional
            embeddings to queries and keys.
        causal_attention: Whether queries can only attend to preceding keys.
        use_bias: Whether projection layers use bias parameters.
        qk_norm: Query and key normalization.
        v_norm: Value normalization.
        use_sdpa: Whether to use scaled dot-product attention.
        rescale_logits: Whether to cancel TimesFM's query pre-scaling with
            inverse-square-root head-dimension scaling.
        device: Device on which to create parameters and buffers.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        use_per_dim_scale: bool = True,
        use_rotary_position_embeddings: bool = True,
        causal_attention: bool = True,
        use_bias: bool = False,
        qk_norm: str = "rms",
        v_norm: str = "none",
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
        else:
            self.query_ln = None
            self.key_ln = None

        if v_norm == "rms":
            self.value_ln = torch.nn.RMSNorm(
                self.head_dim,
                elementwise_affine=False,
                **factory_kwargs,
            )
        else:
            self.value_ln = None

        if use_rotary_position_embeddings:
            self.rotary_position_embedding = RotaryPositionalEmbedding(
                embedding_dims=self.head_dim,
                device=device,
            )
        else:
            self.rotary_position_embedding = None

        if use_per_dim_scale:
            self.per_dim_scale = PerDimScale(
                num_dims=self.head_dim,
                **factory_kwargs,
            )
        else:
            self.per_dim_scale = None

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
        segment_ids: Tensor | None = None,
        segment_pos: Tensor | None = None,
        decode_cache: DecodeCache | None = None,
        patch_mask: Tensor | None = None,
    ) -> tuple[Tensor, DecodeCache | None, Tensor]:
        """Apply multi-head attention.

        Args:
            inputs_q: Inputs with shape ``[B, N, D]``.
            segment_ids: Optional segment identifiers with shape ``[B, N]``.
            segment_pos: Optional rotary positions with shape ``[B, N]``.
            decode_cache: Optional inference-only temporal attention cache.
            patch_mask: Masked patches with shape ``[B, N]``.

        Returns:
            Attention output with shape ``[B, N, D]``, the updated cache, and
            the attention mask with shape ``[B, 1, N, K]``.
        """
        batch_size, num_patches, _ = inputs_q.shape
        if decode_cache is not None and torch.is_grad_enabled():
            raise RuntimeError("Cached decoding requires inference mode")
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

        if decode_cache is None:
            next_index = torch.zeros(
                batch_size,
                dtype=torch.int32,
                device=inputs_q.device,
            )
        else:
            next_index = decode_cache.next_index

        if self.rotary_position_embedding is not None:
            if segment_pos is None:
                position = torch.arange(
                    num_patches,
                    device=inputs_q.device,
                    dtype=torch.int32,
                ).unsqueeze(0) + next_index.unsqueeze(-1)
            else:
                position = segment_pos
            query = self.rotary_position_embedding(query, position)
            key = self.rotary_position_embedding(key, position)

        if self.query_ln is not None:
            query = self.query_ln(query)
        if self.key_ln is not None:
            key = self.key_ln(key)
        if self.per_dim_scale is not None:
            query = self.per_dim_scale(query)
        if self.value_ln is not None:
            value = self.value_ln(value)

        if decode_cache is not None:
            decode_cache = decode_cache.append(
                key,
                value,
                patch_mask=patch_mask,
                segment_ids=segment_ids,
            )
            cache_size = (
                decode_cache.key.size(1)
                if decode_cache.filled is None
                else decode_cache.filled
            )
            key = decode_cache.key[:, :cache_size]
            value = decode_cache.value[:, :cache_size]
            attn_mask = make_attn_mask(
                query_length=num_patches,
                num_all_masked_kv=decode_cache.num_front_masked,
                query_index_offset=next_index,
                kv_length=cache_size,
                causal=self.causal_attention,
            )
            assert decode_cache.patch_mask is not None
            attn_mask = (
                attn_mask
                & ~decode_cache.patch_mask[:, None, None, :cache_size]
            )
            if segment_ids is not None:
                assert decode_cache.segment_ids is not None
                segment_mask = (
                    segment_ids.unsqueeze(2)
                    == decode_cache.segment_ids[:, None, :cache_size]
                ).unsqueeze(1)
                attn_mask = attn_mask & segment_mask
        else:
            attn_mask = make_attn_mask(
                query_length=num_patches,
                num_all_masked_kv=torch.zeros_like(next_index),
                causal=self.causal_attention,
            )
            attn_mask = attn_mask & ~patch_mask[:, None, None, :]
            if segment_ids is not None:
                attn_mask = attn_mask & make_segment_mask(segment_ids)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.use_sdpa:
            scale = 1.0 if self.rescale_logits else math.sqrt(self.head_dim)
            x = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                scale=scale,
            )
        else:
            query = query * math.sqrt(self.head_dim)
            logits = query.matmul(key.transpose(-2, -1))
            if self.rescale_logits:
                logits = logits / math.sqrt(self.head_dim)
            logits = logits.masked_fill(~attn_mask, float("-inf"))
            weights = logits.softmax(dim=-1)
            weights = torch.where(
                attn_mask.any(dim=-1, keepdim=True),
                weights,
                0.0,
            )
            x = weights.matmul(value)

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                num_patches,
                self.in_features,
            )
        )
        return self.out_proj(x), decode_cache, attn_mask


class MixingTransformer(torch.nn.Module):
    """Apply temporal attention, variate attention, and a feed-forward block.

    Args:
        config: Transformer configuration.
        use_variate_attention: Whether to attend across variates.
        device: Device on which to create parameters and buffers.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        config: TransformerConfig,
        use_variate_attention: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.config = config
        self.use_variate_attention = use_variate_attention

        self.pre_seq_attn_ln = torch.nn.RMSNorm(
            config.model_dims,
            **factory_kwargs,
        )
        self.post_seq_attn_ln = torch.nn.RMSNorm(
            config.model_dims,
            **factory_kwargs,
        )
        rescale_logits = not config.use_memory_efficient_attention
        self.seq_attn = MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.model_dims,
            use_per_dim_scale=True,
            use_rotary_position_embeddings=config.use_rope_seq,
            qk_norm=config.qk_norm,
            v_norm=config.v_norm,
            causal_attention=config.causal_attention,
            use_bias=config.use_bias,
            use_sdpa=config.use_sdpa,
            rescale_logits=rescale_logits,
            **factory_kwargs,
        )

        if use_variate_attention:
            self.pre_var_attn_ln = torch.nn.RMSNorm(
                config.model_dims,
                **factory_kwargs,
            )
            self.post_var_attn_ln = torch.nn.RMSNorm(
                config.model_dims,
                **factory_kwargs,
            )
            self.var_attn = MultiHeadAttention(
                num_heads=config.num_heads,
                in_features=config.model_dims,
                use_per_dim_scale=True,
                use_rotary_position_embeddings=config.use_rope_var,
                qk_norm=config.qk_norm,
                v_norm=config.v_norm,
                causal_attention=False,
                use_bias=config.use_bias,
                use_sdpa=config.use_sdpa,
                rescale_logits=rescale_logits,
                **factory_kwargs,
            )

        self.pre_ff_ln = torch.nn.RMSNorm(
            config.model_dims,
            **factory_kwargs,
        )
        self.post_ff_ln = torch.nn.RMSNorm(
            config.model_dims,
            **factory_kwargs,
        )
        self.ff0 = torch.nn.Linear(
            config.model_dims,
            config.hidden_dims,
            bias=config.use_bias,
            **factory_kwargs,
        )
        self.ff1 = torch.nn.Linear(
            config.hidden_dims,
            config.model_dims,
            bias=config.use_bias,
            **factory_kwargs,
        )
        self.activation = get_activation_fn(config.ff_activation)

    def forward(
        self,
        input_embeddings: Tensor,
        patch_mask: Tensor,
        segment_ids: Tensor | None = None,
        segment_pos: Tensor | None = None,
        decode_cache: DecodeCache | None = None,
        var_segment_pos: Tensor | None = None,
    ) -> tuple[Tensor, DecodeCache | None, Tensor]:
        """Apply one mixing transformer layer.

        Args:
            input_embeddings: Inputs with shape ``[B, V, N, D]``.
            patch_mask: Masked patches with shape ``[B, V, N]``.
            segment_ids: Optional segment identifiers with shape ``[B, N]``.
            segment_pos: Optional temporal positions with shape ``[B, N]``.
            decode_cache: Optional temporal attention cache.
            var_segment_pos: Optional variate positions with shape
                ``[B * N, V]``.

        Returns:
            Output with shape ``[B, V, N, D]``, the updated temporal cache,
            and the temporal attention mask.
        """
        batch_size, num_variates, num_patches, model_dims = (
            input_embeddings.shape
        )

        seq_input = self.pre_seq_attn_ln(input_embeddings).reshape(
            batch_size * num_variates,
            num_patches,
            model_dims,
        )
        seq_patch_mask = patch_mask.reshape(
            batch_size * num_variates,
            num_patches,
        )

        seq_segment_ids = None
        if segment_ids is not None:
            seq_segment_ids = (
                segment_ids[:, None]
                .expand(
                    batch_size,
                    num_variates,
                    num_patches,
                )
                .reshape(batch_size * num_variates, num_patches)
            )

        seq_segment_pos = None
        if segment_pos is not None:
            seq_segment_pos = (
                segment_pos[:, None]
                .expand(
                    batch_size,
                    num_variates,
                    num_patches,
                )
                .reshape(batch_size * num_variates, num_patches)
            )

        seq_output, decode_cache, seq_attn_mask = self.seq_attn(
            seq_input,
            segment_ids=seq_segment_ids,
            segment_pos=seq_segment_pos,
            decode_cache=decode_cache,
            patch_mask=seq_patch_mask,
        )
        seq_output = seq_output.view(
            batch_size,
            num_variates,
            num_patches,
            model_dims,
        )
        hidden = self.post_seq_attn_ln(seq_output) + input_embeddings

        if self.use_variate_attention:
            var_input = self.pre_var_attn_ln(hidden)
            var_input = var_input.permute(0, 2, 1, 3).reshape(
                batch_size * num_patches,
                num_variates,
                model_dims,
            )
            var_patch_mask = patch_mask.permute(0, 2, 1).reshape(
                batch_size * num_patches,
                num_variates,
            )
            var_output, _, _ = self.var_attn(
                var_input,
                segment_pos=var_segment_pos,
                decode_cache=None,
                patch_mask=var_patch_mask,
            )
            var_output = var_output.view(
                batch_size,
                num_patches,
                num_variates,
                model_dims,
            ).permute(0, 2, 1, 3)
            hidden = self.post_var_attn_ln(var_output) + hidden

        ff_output = self.ff1(self.activation(self.ff0(self.pre_ff_ln(hidden))))
        return (
            self.post_ff_ln(ff_output) + hidden,
            decode_cache,
            seq_attn_mask,
        )


class StackedMixingTransformer(torch.nn.Module):
    """Apply a stack of TimesFM-3 mixing transformer layers.

    Args:
        config: Stacked transformer configuration.
        use_variate_attention: Whether to attend across variates.
        device: Device on which to create parameters and buffers.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        config: StackedTransformersConfig,
        use_variate_attention: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList(
            [
                MixingTransformer(
                    config=config.transformer,
                    use_variate_attention=use_variate_attention,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(
        self,
        input_embeddings: Tensor,
        patch_mask: Tensor,
        segment_ids: Tensor | None = None,
        segment_pos: Tensor | None = None,
        decode_cache: list[DecodeCache | None] | None = None,
        var_segment_pos: Tensor | None = None,
    ) -> tuple[
        Tensor,
        list[DecodeCache | None],
        list[Tensor],
    ]:
        """Apply the mixing transformer stack.

        Args:
            input_embeddings: Inputs with shape ``[B, V, N, D]``.
            patch_mask: Masked patches with shape ``[B, V, N]``.
            segment_ids: Optional segment identifiers with shape ``[B, N]``.
            segment_pos: Optional temporal positions with shape ``[B, N]``.
            decode_cache: Optional cache for every transformer layer.
            var_segment_pos: Optional variate positions with shape
                ``[B * N, V]``.

        Returns:
            Output with shape ``[B, V, N, D]``, the cache result and temporal
            attention mask from every layer.
        """
        caches: list[DecodeCache | None]
        if decode_cache is None:
            caches = [None] * len(self.layers)
        else:
            caches = decode_cache

        output = input_embeddings
        new_caches: list[DecodeCache | None] = []
        attn_masks = []
        for index, layer in enumerate(self.layers):
            output, layer_cache, layer_mask = layer(
                output,
                patch_mask,
                segment_ids,
                segment_pos,
                caches[index],
                var_segment_pos,
            )
            new_caches.append(layer_cache)
            attn_masks.append(layer_mask)

        return output, new_caches, attn_masks
