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
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList, Parameter

from sdm.nn import SDPA, RotaryEmbedding


class _RMSNorm(torch.nn.Module):
    """TabFM RMSNorm with a fully float32 normalization path."""

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Parameter(
            torch.ones(channels, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.square().mean(dim=-1, keepdim=True)
        x = x * (variance + self.eps).rsqrt()
        return (x * self.weight.float()).to(dtype)


class MultiheadAttention(torch.nn.Module):
    """Apply TabFM v1.0.0 multi-head attention.

    Query and key heads are independently RMS-normalized. Queries then receive
    a learned positive scale per head channel before attention is evaluated
    with an explicit logit scale of ``1.0``.

    Args:
        channels: Number of input and output channels.
        num_heads: Number of attention heads. Must divide ``channels``.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
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

        self.q_proj = Linear(channels, channels, **factory_kwargs)
        self.k_proj = Linear(channels, channels, **factory_kwargs)
        self.v_proj = Linear(channels, channels, **factory_kwargs)
        self.out_proj = Linear(channels, channels, **factory_kwargs)
        self.query_ln = _RMSNorm(
            self.head_channels,
            eps=1e-6,
            **factory_kwargs,
        )
        self.key_ln = _RMSNorm(
            self.head_channels,
            eps=1e-6,
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
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        rope: RotaryEmbedding | None = None,
    ) -> Tensor:
        """Apply attention to projected, normalized, and scaled inputs.

        Args:
            query: Query tensor with shape ``[..., Q, D]``. ``Q`` is the query
                sequence length and ``D`` is ``channels``.
            key: Key tensor with shape ``[..., KV, D]``. ``KV`` is the
                key/value sequence length.
            value: Value tensor with shape ``[..., KV, D]``.
            attn_mask: Boolean attention mask with shape ``[..., Q, KV]``.
                TabFM's ``[..., 1, Q, KV]`` singleton-head layout is also
                accepted.
            rope: Optional shared interleaved rotary embedding applied to
                projected queries and keys.

        Returns:
            Tensor with shape ``[..., Q, D]``.
        """
        input_rank = max(query.dim(), key.dim(), value.dim())
        if (
            attn_mask is not None
            and attn_mask.dim() == input_rank + 1
            and attn_mask.size(-3) == 1
        ):
            attn_mask = attn_mask.squeeze(-3)

        head_shape = (self.num_heads, self.head_channels)
        query = self.q_proj(query).unflatten(-1, head_shape)
        key = self.k_proj(key).unflatten(-1, head_shape)
        value = self.v_proj(value).unflatten(-1, head_shape)

        if rope is not None:
            query = rope(query)
            key = rope(key)

        query = self.query_ln(query)
        key = self.key_ln(key)
        scale = (
            1.442695041
            / math.sqrt(self.head_channels)
            * F.softplus(self.per_dim_scale.float())
        )
        query = query * scale.to(query.dtype)

        output = self.sdpa(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )
        return self.out_proj(output.flatten(-2))


class _MultiheadAttentionBlock(torch.nn.Module):
    """Apply a TabFM v1.0.0 residual attention and SwiGLU block.

    Args:
        channels: Number of input and output channels.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of the SwiGLU feed-forward layer.
        ffn_chunk_size: Optional maximum number of flattened tokens processed
            by the feed-forward layer at once.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        feedforward_channels: int,
        ffn_chunk_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if feedforward_channels <= 0:
            raise ValueError("feedforward_channels must be positive")
        if ffn_chunk_size is not None and ffn_chunk_size <= 0:
            raise ValueError("ffn_chunk_size must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.attn = MultiheadAttention(
            channels=channels,
            num_heads=num_heads,
            **factory_kwargs,
        )
        self.pre_attn_ln = _RMSNorm(channels, eps=1e-6, **factory_kwargs)
        self.post_attn_ln = _RMSNorm(channels, eps=1e-6, **factory_kwargs)
        self.pre_ff_ln = _RMSNorm(channels, eps=1e-6, **factory_kwargs)
        self.post_ff_ln = _RMSNorm(channels, eps=1e-6, **factory_kwargs)
        self.linear1 = Linear(channels, feedforward_channels, **factory_kwargs)
        self.linear1_gate = Linear(
            channels,
            feedforward_channels,
            **factory_kwargs,
        )
        self.linear2 = Linear(feedforward_channels, channels, **factory_kwargs)
        self.ffn_chunk_size = ffn_chunk_size

    def _feedforward_impl(self, x: Tensor) -> Tensor:
        normalized = self.pre_ff_ln(x)
        hidden = F.silu(self.linear1_gate(normalized))
        hidden = hidden * self.linear1(normalized)
        return self.post_ff_ln(self.linear2(hidden))

    def _feedforward(self, x: Tensor) -> Tensor:
        if self.ffn_chunk_size is None:
            return self._feedforward_impl(x)

        # Flatten independent batch and sequence axes into tokens.
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        if x.size(0) == 0:
            return self._feedforward_impl(x).view(shape)

        output = x.new_empty(x.size(0), self.linear2.out_features)
        for start in range(0, x.size(0), self.ffn_chunk_size):
            stop = start + self.ffn_chunk_size
            output[start:stop].copy_(self._feedforward_impl(x[start:stop]))
        return output.view(shape)

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        rope: RotaryEmbedding | None = None,
    ) -> Tensor:
        """Run attention and feed-forward; omitted ``value`` uses ``query``."""
        key = query if key is None else key
        value = query if value is None else value
        attention = self.attn(
            query=self.pre_attn_ln(query),
            key=self.pre_attn_ln(key),
            value=self.pre_attn_ln(value),
            attn_mask=attn_mask,
            rope=rope,
        )
        output = query + self.post_attn_ln(attention)
        return output + self._feedforward(output)


class _Encoder(torch.nn.Module):
    """Stack TabFM v1.0.0 multi-head attention blocks.

    Args:
        num_blocks: Number of attention blocks.
        channels: Number of input and output channels.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of each SwiGLU feed-forward layer.
        rope_theta: Rotary frequency base. ``None`` disables rotary embedding.
        ffn_chunk_size: Optional maximum number of flattened tokens processed
            by each feed-forward layer at once.
        device: Device on which to create parameters.
        dtype: Dtype of parameters.
    """

    def __init__(
        self,
        num_blocks: int,
        channels: int,
        num_heads: int,
        feedforward_channels: int,
        rope_theta: float | None = 100_000.0,
        ffn_chunk_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if num_heads <= 0 or channels % num_heads != 0:
            raise ValueError("num_heads must be positive and divide channels")
        if rope_theta is not None and rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.rope: RotaryEmbedding | None = None
        if rope_theta is not None:
            self.rope = RotaryEmbedding(
                channels=channels // num_heads,
                layout="interleaved",
                theta=rope_theta,
                requires_grad=False,
                **factory_kwargs,
            )
        self.blocks = ModuleList(
            _MultiheadAttentionBlock(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                ffn_chunk_size=ffn_chunk_size,
                **factory_kwargs,
            )
            for _ in range(num_blocks)
        )

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Encode a sequence with the stacked attention blocks."""
        for block in self.blocks:
            x = block(query=x, attn_mask=attn_mask, rope=self.rope)
        return x
