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
#
# Modified for the structured-data-models package.

import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList, Parameter

from sdm.cache import Cache, KVCacheEntry
from sdm.models.tabfm._utils import _get_activation
from sdm.nn.attention import SDPA
from sdm.nn.rope import apply_rotary_embedding


class RMSNorm(torch.nn.Module):
    """Apply root-mean-square normalization over the final dimension.

    The variance, normalization, and learned scale are evaluated in
    ``torch.float32`` before the result is cast back to the input dtype.
    This matches the released TabFM PyTorch checkpoint implementation.

    Args:
        channels: Size of the final input dimension.
        epsilon: Non-negative value added before the reciprocal square root.
        device: Device on which to create the learnable scale.
        dtype: Dtype of the learnable scale.
    """

    def __init__(
        self,
        channels: int,
        epsilon: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        self.epsilon = epsilon
        self.weight = Parameter(
            torch.ones(channels, device=device, dtype=dtype)
        )

    def forward(self, input: Tensor) -> Tensor:
        """Normalize ``input`` over its final dimension.

        Args:
            input: Input tensor with shape ``[..., C]``, where ``C`` is
                ``channels``.

        Returns:
            Tensor with shape ``[..., C]`` and the input dtype.
        """
        input_dtype = input.dtype
        input_float = input.float()
        variance = input_float.square().mean(dim=-1, keepdim=True)
        normalized = input_float * (variance + self.epsilon).rsqrt()
        return (normalized * self.weight.float()).to(input_dtype)


class RotaryEmbedding(torch.nn.Module):
    """Apply interleaved rotary position embeddings to attention heads.

    The inverse-frequency buffer is part of the state dict because released
    TabFM checkpoints store the frequencies used during training. Loading that
    buffer avoids numerical drift from recomputing it in another dtype.

    Args:
        channels: Number of channels in each attention head. Must be even.
        theta: Base used to initialize inverse frequencies.
        device: Device on which to create the frequency buffer.
        dtype: Dtype of the frequency buffer. Defaults to ``torch.float32``
            when omitted.
    """

    freqs: Tensor

    def __init__(
        self,
        channels: int,
        theta: float = 100_000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0 or channels % 2 != 0:
            raise ValueError("channels must be a positive even number")
        if theta <= 0:
            raise ValueError("theta must be positive")

        channel_index = torch.arange(
            0,
            channels,
            2,
            device=device,
            dtype=torch.float32,
        )
        frequencies = 1.0 / (theta ** (channel_index / channels))
        if dtype is not None:
            frequencies = frequencies.to(dtype)
        self.register_buffer("freqs", frequencies)

    def forward(self, input: Tensor) -> Tensor:
        """Rotate query or key heads along their sequence dimension.

        Args:
            input: Query or key tensor with shape ``[..., S, H, C]``. ``S`` is
                the sequence length, ``H`` is the number of heads, and ``C`` is
                the number of channels per head.

        Returns:
            Tensor with shape ``[..., S, H, C]``.
        """
        if input.size(-1) != 2 * self.freqs.numel():
            raise ValueError(
                "input head channels do not match rotary frequencies"
            )
        return apply_rotary_embedding(
            input,
            self.freqs,
            layout="interleaved",
        )


class MultiheadAttention(torch.nn.Module):
    """TabFM multi-head attention with normalized and scaled queries.

    Query and key heads are independently RMS-normalized. Queries additionally
    receive a learned positive scale per head channel before scaled dot-product
    attention is evaluated with an explicit scale of ``1.0``.

    Args:
        channels: Number of input and output channels.
        num_heads: Number of attention heads.
        rope_theta: Rotary embedding base. When omitted, rotary embeddings are
            disabled.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        rope_theta: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if num_heads <= 0 or channels % num_heads != 0:
            raise ValueError("num_heads must be positive and divide channels")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.rope_theta = rope_theta

        self.q_proj = Linear(channels, channels, **factory_kwargs)
        self.k_proj = Linear(channels, channels, **factory_kwargs)
        self.v_proj = Linear(channels, channels, **factory_kwargs)
        self.out_proj = Linear(channels, channels, **factory_kwargs)
        self.query_ln = RMSNorm(
            self.head_channels,
            **factory_kwargs,
        )
        self.key_ln = RMSNorm(
            self.head_channels,
            **factory_kwargs,
        )
        self.per_dim_scale = Parameter(
            torch.zeros(self.head_channels, **factory_kwargs)
        )
        self.sdpa = SDPA(
            channels=self.head_channels,
            num_query_heads=num_heads,
            scale=1.0,
            **factory_kwargs,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor | KVCacheEntry,
        value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        rope: RotaryEmbedding | None = None,
        return_key_value: bool = False,
    ) -> Tensor | tuple[Tensor, KVCacheEntry]:
        """Apply attention from ``query`` to ``key`` and ``value``.

        Args:
            query: Query tensor with shape ``[..., Q, D]``.
            key: Key tensor with shape ``[..., KV, D]`` or cached projected
                keys and values.
            value: Value tensor with shape ``[..., KV, D]``. Required when
                ``key`` is a tensor.
            attn_mask: Boolean or additive attention mask broadcastable to
                ``[..., H, Q, KV]``.
            rope: Shared rotary embedding. Used only when ``rope_theta`` was
                configured. If omitted in that case, frequencies are computed
                from ``rope_theta`` for this call.
            return_key_value: Whether to return projected keys and values for
                later replay.

        Returns:
            Tensor with shape ``[..., Q, D]``. When ``return_key_value`` is
            true, also returns a :class:`~sdm.cache.KVCacheEntry`.
        """
        *batch_shape, query_length, channels = query.shape
        query = self.q_proj(query).view(
            *batch_shape,
            query_length,
            self.num_heads,
            self.head_channels,
        )

        cached = isinstance(key, KVCacheEntry)
        if cached:
            if value is not None:
                raise ValueError("value must be omitted with cached keys")
            projected_key = key.key
            projected_value = key.value
            if projected_key.shape[:-3] != tuple(batch_shape):
                raise ValueError(
                    "query and cached key must have equal batch dimensions"
                )
            if projected_value.shape != projected_key.shape:
                raise ValueError("cached key and value shapes must match")
            if projected_key.shape[-2:] != (
                self.num_heads,
                self.head_channels,
            ):
                raise ValueError("cached key head shape is incompatible")
        else:
            if value is None:
                raise ValueError("value is required when key is a tensor")
            key_length = key.size(-2)
            if key.shape[:-2] != tuple(batch_shape):
                raise ValueError(
                    "query and key must have equal batch dimensions"
                )
            if value.shape[:-2] != tuple(batch_shape):
                raise ValueError(
                    "query and value must have equal batch dimensions"
                )
            if key.size(-1) != channels or value.size(-1) != channels:
                raise ValueError("query, key, and value channels must match")
            if value.size(-2) != key_length:
                raise ValueError("key and value sequence lengths must match")
            projected_key = self.k_proj(key).view(
                *batch_shape,
                key_length,
                self.num_heads,
                self.head_channels,
            )
            projected_value = self.v_proj(value).view(
                *batch_shape,
                key_length,
                self.num_heads,
                self.head_channels,
            )

        if self.rope_theta is not None:
            if rope is not None:
                query = rope(query)
                if not cached:
                    projected_key = rope(projected_key)
            else:
                index = torch.arange(
                    0,
                    self.head_channels,
                    2,
                    device=query.device,
                    dtype=torch.float32,
                )
                frequencies = 1.0 / (
                    self.rope_theta ** (index / self.head_channels)
                )
                query = apply_rotary_embedding(
                    query,
                    frequencies,
                    layout="interleaved",
                )
                if not cached:
                    projected_key = apply_rotary_embedding(
                        projected_key,
                        frequencies,
                        layout="interleaved",
                    )

        query = self.query_ln(query)
        if not cached:
            projected_key = self.key_ln(projected_key)
        scale = (
            1.442695041
            / math.sqrt(self.head_channels)
            * F.softplus(self.per_dim_scale.float())
        )
        query = query * scale.to(query.dtype)

        if (
            attn_mask is not None
            and attn_mask.dim() >= 3
            and attn_mask.size(-3) == 1
        ):
            attn_mask = attn_mask.squeeze(-3)
        output = self.sdpa(
            query=query,
            key=projected_key,
            value=projected_value,
            attn_mask=attn_mask,
        )
        output = output.reshape(
            *batch_shape,
            query_length,
            channels,
        )
        output = self.out_proj(output)
        if return_key_value:
            return output, KVCacheEntry(
                key=projected_key,
                value=projected_value,
            )
        return output


class MultiheadAttentionBlock(torch.nn.Module):
    """TabFM residual attention and feed-forward block.

    Args:
        channels: Number of input and output channels.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of the feed-forward network.
        activation: Feed-forward activation. ``"swiglu"`` uses a learned SiLU
            gate; ``"relu"``, ``"gelu"``, and ``"silu"`` use one projection.
        rope_theta: Rotary embedding base. When omitted, rotary embeddings are
            disabled.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        feedforward_channels: int,
        activation: str = "swiglu",
        rope_theta: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if feedforward_channels <= 0:
            raise ValueError("feedforward_channels must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.attn = MultiheadAttention(
            channels=channels,
            num_heads=num_heads,
            rope_theta=rope_theta,
            **factory_kwargs,
        )
        self.pre_attn_ln = RMSNorm(channels, **factory_kwargs)
        self.post_attn_ln = RMSNorm(channels, **factory_kwargs)
        self.pre_ff_ln = RMSNorm(channels, **factory_kwargs)
        self.post_ff_ln = RMSNorm(channels, **factory_kwargs)
        self.swiglu = activation == "swiglu"
        self.linear1 = Linear(
            channels,
            feedforward_channels,
            **factory_kwargs,
        )
        if self.swiglu:
            self.linear1_gate: Linear | None = Linear(
                channels,
                feedforward_channels,
                **factory_kwargs,
            )
            self.activation = F.silu
        else:
            self.linear1_gate = None
            self.activation = _get_activation(activation)
        self.linear2 = Linear(
            feedforward_channels,
            channels,
            **factory_kwargs,
        )
        self.ffn_chunk_size: int | None = None

    def _feedforward_impl(self, input: Tensor) -> Tensor:
        normalized = self.pre_ff_ln(input)
        hidden = self.linear1(normalized)
        if self.swiglu:
            assert self.linear1_gate is not None
            hidden = self.activation(self.linear1_gate(normalized)) * hidden
        else:
            hidden = self.activation(hidden)
        return self.post_ff_ln(self.linear2(hidden))

    def _feedforward(self, input: Tensor) -> Tensor:
        if self.ffn_chunk_size is None:
            return self._feedforward_impl(input)

        shape = input.shape
        flattened = input.reshape(-1, shape[-1])
        output = flattened.new_empty(
            flattened.size(0),
            self.linear2.out_features,
        )
        # Chunking avoids materializing the full expanded activation. This loop
        # is intentional because each token chunk is independent.
        for start in range(0, flattened.size(0), self.ffn_chunk_size):
            stop = start + self.ffn_chunk_size
            output[start:stop] = self._feedforward_impl(flattened[start:stop])
        return output.view(shape)

    def forward(
        self,
        query: Tensor,
        key: Tensor | KVCacheEntry | None = None,
        value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        rope: RotaryEmbedding | None = None,
        return_key_value: bool = False,
    ) -> Tensor | tuple[Tensor, KVCacheEntry]:
        """Apply residual attention followed by a residual feed-forward layer.

        Args:
            query: Query tensor with shape ``[..., Q, D]``.
            key: Optional key tensor with shape ``[..., KV, D]`` or cached
                projected keys and values. Defaults to ``query``.
            value: Optional value tensor with shape ``[..., KV, D]``. Defaults
                to ``query``.
            attn_mask: Boolean or additive attention mask broadcastable to
                ``[..., H, Q, KV]``.
            rope: Optional shared rotary embedding.
            return_key_value: Whether to return projected keys and values.

        Returns:
            Tensor with shape ``[..., Q, D]`` and, when requested, projected
            keys and values.
        """
        key = query if key is None else key
        if value is None and isinstance(key, Tensor):
            value = key
        normalized_key: Tensor | KVCacheEntry
        normalized_value: Tensor | None
        if isinstance(key, KVCacheEntry):
            normalized_key = key
            normalized_value = None
        else:
            normalized_key = self.pre_attn_ln(key)
            assert value is not None
            normalized_value = self.pre_attn_ln(value)
        attention_result = self.attn(
            self.pre_attn_ln(query),
            normalized_key,
            normalized_value,
            attn_mask=attn_mask,
            rope=rope,
            return_key_value=return_key_value,
        )
        if return_key_value:
            attention, key_value = cast(
                tuple[Tensor, KVCacheEntry],
                attention_result,
            )
        else:
            attention = cast(Tensor, attention_result)
        attention = self.post_attn_ln(attention)
        output = query + attention
        output = output + self._feedforward(output)
        if return_key_value:
            return output, key_value
        return output


class Encoder(torch.nn.Module):
    """Stack TabFM multi-head attention blocks with one shared RoPE buffer.

    Args:
        num_blocks: Number of attention blocks.
        channels: Number of input and output channels.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of each feed-forward network.
        activation: Feed-forward activation used by every block.
        rope_theta: Shared rotary embedding base. When omitted, rotary
            embeddings are disabled.
        device: Device on which to create parameters and buffers.
        dtype: Dtype of parameters and buffers.
    """

    def __init__(
        self,
        num_blocks: int,
        channels: int,
        num_heads: int,
        feedforward_channels: int,
        activation: str = "swiglu",
        rope_theta: float | None = 100_000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.rope: RotaryEmbedding | None = None
        if rope_theta is not None:
            self.rope = RotaryEmbedding(
                channels=channels // num_heads,
                theta=rope_theta,
                **factory_kwargs,
            )
        self.blocks = ModuleList(
            MultiheadAttentionBlock(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                activation=activation,
                rope_theta=rope_theta,
                **factory_kwargs,
            )
            for _ in range(num_blocks)
        )

    def forward(
        self,
        input: Tensor,
        attn_mask: Tensor | None = None,
        cache: Cache | None = None,
        cache_prefix: str = "encoder",
    ) -> Tensor:
        """Encode a sequence with the stacked attention blocks.

        Args:
            input: Input tensor with shape ``[..., S, D]``.
            attn_mask: Boolean or additive attention mask broadcastable to
                ``[..., H, S, S]``.
            cache: Optional record/replay cache for context keys and values.
                Caching is supported only for RoPE-disabled encoders.
            cache_prefix: Key namespace used within ``cache``.

        Returns:
            Tensor with shape ``[..., S, D]``.
        """
        if cache is not None and self.rope is not None:
            raise ValueError("encoder caching requires RoPE to be disabled")
        for index, block in enumerate(self.blocks):
            key = f"{cache_prefix}.block{index}"
            key_value = (
                cast(KVCacheEntry, cache[key])
                if cache is not None and cache.is_replaying
                else None
            )
            result = block(
                input,
                key=key_value,
                attn_mask=attn_mask,
                rope=self.rope,
                return_key_value=cache is not None and cache.is_recording,
            )
            if cache is not None and cache.is_recording:
                input, cache[key] = cast(
                    tuple[Tensor, KVCacheEntry],
                    result,
                )
            else:
                input = cast(Tensor, result)
        return input
