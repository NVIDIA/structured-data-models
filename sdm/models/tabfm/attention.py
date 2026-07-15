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
from torch.nn import Linear, Parameter, RMSNorm

from sdm.nn import SDPA, RotaryEmbedding


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
        self.query_ln = RMSNorm(
            self.head_channels,
            eps=1e-6,
            **factory_kwargs,
        )
        self.key_ln = RMSNorm(
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
            rope: Optional shared interleaved rotary embedding applied to
                projected queries and keys.

        Returns:
            Tensor with shape ``[..., Q, D]``.
        """
        if query.dim() < 2 or key.dim() < 2 or value.dim() < 2:
            raise ValueError("query, key, and value must have at least 2 dims")
        if query.shape[:-2] != key.shape[:-2]:
            raise ValueError("query and key must have equal batch dimensions")
        if query.shape[:-2] != value.shape[:-2]:
            raise ValueError(
                "query and value must have equal batch dimensions"
            )
        if key.size(-2) != value.size(-2):
            raise ValueError("key and value sequence lengths must match")
        channels = self.q_proj.in_features
        if (
            query.size(-1) != channels
            or key.size(-1) != channels
            or value.size(-1) != channels
        ):
            raise ValueError("query, key, and value channels must match")

        *batch_shape, query_length, _ = query.shape
        key_length = key.size(-2)
        query = self.q_proj(query).view(
            *batch_shape,
            query_length,
            self.num_heads,
            self.head_channels,
        )
        key = self.k_proj(key).view(
            *batch_shape,
            key_length,
            self.num_heads,
            self.head_channels,
        )
        value = self.v_proj(value).view(
            *batch_shape,
            key_length,
            self.num_heads,
            self.head_channels,
        )

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

        if (
            attn_mask is not None
            and attn_mask.dim() >= 3
            and attn_mask.size(-3) == 1
        ):
            attn_mask = attn_mask.squeeze(-3)
        output = self.sdpa(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )
        output = output.reshape(*batch_shape, query_length, channels)
        return self.out_proj(output)
