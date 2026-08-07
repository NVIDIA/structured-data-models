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
from torch.nn import Linear, Parameter

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
