"""Attention modules for structured tensor models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GELU, Linear, Sequential

if TYPE_CHECKING:
    from schemafm.nn import RotaryEmbedding


class QASSMax(torch.nn.Module):
    r"""Learnable query scaler for query-aware scalable softmax (QASSMax).

    This scaling method was introduced in the `"TabICLv2: A better, faster,
    scalable, and open tabular foundation model"
    <https://arxiv.org/abs/2602.11139>`_ paper as a temperature-scaling method
    for attention.

    For a query tensor ``q`` and key length ``n``, this module returns a scaled
    query

    .. math::

        \tilde{q}_{hi} = q_{hi} \cdot
          \mathrm{MLP}_{\mathrm{base}}(\log n)_{hi} \cdot
          \left(1 + \tanh(\mathrm{MLP}_{\mathrm{gate}}(q_h)_i)\right),

    where ``h`` indexes attention heads, ``i`` indexes head channels,
    ``MLP_base`` maps the log key length to per-head, per-channel scale
    factors, and ``MLP_gate`` maps each per-head query vector to a bounded
    query-dependent gate. In this implementation, ``n`` is clamped to at least
    1 before taking ``log``. The gate is initialized as identity modulation by
    zero-initializing the final layer of ``MLP_gate``.

    Multiplying the query scales the subsequent attention logits while keeping
    the attention computation compatible with standard softmax kernels. The
    length-dependent factor counteracts attention fading as the number of keys
    grows, while the query-dependent gate lets the scale vary across queries.

    Args:
        channels: The number of channels per attention head.
        num_heads: The number of attention heads.
        hidden_channels: The hidden width of the scale and gate MLPs.
        device: The device to use for module parameters.
        dtype: The dtype to use for module parameters.

    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        hidden_channels: int = 64,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.scale = Sequential(
            Linear(1, hidden_channels, **factory_kwargs),
            GELU(),
            Linear(hidden_channels, num_heads * channels, **factory_kwargs),
        )
        self.gate = Sequential(
            Linear(channels, hidden_channels, **factory_kwargs),
            GELU(),
            Linear(hidden_channels, channels, **factory_kwargs),
        )

        torch.nn.init.zeros_(cast(Linear, self.gate[-1]).weight)
        torch.nn.init.zeros_(cast(Linear, self.gate[-1]).bias)

    def forward(
        self,
        query: Tensor,  # [..., S, H, C]
        key_len: Tensor | int,  # [..., 1] or [..., S] or scalar
    ) -> Tensor:  # [..., S, H, C]
        r"""Forward pass of :class:`QASSMax`.

        Args:
            query: The query tensor to scale, with shape ``[..., S, H, C]``.
                ``S`` is the query sequence length, ``H`` is the number of
                attention heads, and ``C`` is the channels per head.
            key_len: The number of valid keys used to scale each query.

                Supported forms are:

                * ``int``: one shared key length for every query.
                * tensor shaped ``[..., 1]``: one key length per batch/context
                  item, shared across all query positions in ``S``.
                * tensor shaped ``[..., S]``: one key length per query
                  position, such as the result of reducing an attention mask
                  over its key dimension.

        Returns:
            The scaled query tensor.

        """
        if isinstance(key_len, Tensor):
            log_key_len = key_len.float().clamp(min=1.0).log().to(query.dtype)
        else:
            one = query.new_ones((), dtype=torch.float32)
            log_key_len = (one * key_len).clamp(min=1.0).log().to(query.dtype)
            log_key_len = log_key_len.view([1] * (query.dim() - 2))
        scale = self.scale(log_key_len.unsqueeze(-1))  # [..., S or 1, H * C]
        scale = scale.unflatten(-1, (query.size(-2), query.size(-1)))
        gate = 1 + self.gate(query).tanh()
        return query * scale * gate


class SDPA(torch.nn.Module):
    r"""Scaled dot-product attention wrapper for ``[..., S, H, C]`` tensors."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        qassmax: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.qassmax: QASSMax | None = None
        if qassmax:
            self.qassmax = QASSMax(
                channels=channels,
                num_heads=num_heads,
                **factory_kwargs,
            )

    def forward(
        self,
        query: Tensor,  # [..., Q, H, C]
        key: Tensor,  # [..., KV, H, C]
        value: Tensor,  # [..., KV, H, C]
        seqused_key_value: Tensor | None = None,  # [...]
        attn_mask: Tensor | None = None,  # [..., Q, KV]
    ) -> Tensor:  # [..., Q, H, C]
        r"""Apply scaled dot-product attention.

        Args:
            query: Query tensor with shape ``[..., Q, H, C]``.
            key: Key tensor with shape ``[..., KV, H, C]``.
            value: Value tensor with shape ``[..., KV, H, C]``.
            seqused_key_value: Optional valid key/value lengths with shape
                ``[...]`` and dtype ``torch.int32``.
            attn_mask: Optional boolean attention mask with shape
                ``[..., Q, KV]``. Entries set to ``True`` participate in
                attention.

        Returns:
            The attention output with shape ``[..., Q, H, C]``.

        """
        if attn_mask is not None and seqused_key_value is not None:
            raise ValueError(
                "Cannot pass both `attn_mask` and `seqused_key_value`"
            )

        if (
            seqused_key_value is not None
            and seqused_key_value.dtype != torch.int32
        ):
            raise ValueError("`seqused_key_value` must have dtype torch.int32")
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            raise ValueError("`attn_mask` must have dtype torch.bool")

        if self.qassmax is not None:
            if seqused_key_value is not None:
                key_len = seqused_key_value.unsqueeze(-1)
            elif attn_mask is not None and attn_mask.size(-1) > 1:
                key_len = attn_mask.sum(dim=-1)
            else:
                key_len = key.size(-3)
            query = self.qassmax(query, key_len=key_len)

        batch_shapes = [query.size()[:-3], key.size()[:-3], value.size()[:-3]]
        if seqused_key_value is not None:
            batch_shapes.append(seqused_key_value.size())
        if attn_mask is not None:
            batch_shapes.append(attn_mask.size()[:-2])
        batch_shape = torch.broadcast_shapes(*batch_shapes)

        # Broadcast and flatten batch dimensions => [B, S, H, C].
        query_size = query.size()[-3:]
        key_size = key.size()[-3:]
        value_size = value.size()[-3:]
        query = query.expand(batch_shape + query_size).reshape(-1, *query_size)
        key = key.expand(batch_shape + key_size).reshape(-1, *key_size)
        value = value.expand(batch_shape + value_size).reshape(-1, *value_size)

        if attn_mask is not None:
            attn_mask = attn_mask.expand(batch_shape + attn_mask.size()[-2:])
            attn_mask = attn_mask.reshape(-1, *attn_mask.size()[-2:])

        if seqused_key_value is not None:
            seqused_key_value = seqused_key_value.expand(batch_shape)
            seqused_key_value = seqused_key_value.reshape(-1).unsqueeze(-1)
            key_index = torch.arange(key.size(-3), device=key.device)
            attn_mask = key_index.unsqueeze(0) < seqused_key_value
            attn_mask = attn_mask.unsqueeze(-2).expand(-1, query.size(-3), -1)

        out = F.scaled_dot_product_attention(
            query=query.transpose(-3, -2),  # [B, H, Q, C],
            key=key.transpose(-3, -2),  # [B, H, KV, C],
            value=value.transpose(-3, -2),  # [B, H, KV, C],
            attn_mask=attn_mask.unsqueeze(-3)  # [B, 1, Q, KV]
            if attn_mask is not None
            else None,
        ).transpose(-3, -2)  # [B, Q, H, C]

        return out.view(batch_shape + out.size()[-3:])  # [..., Q, H, C]


class MultiHeadAttention(torch.nn.Module):
    r"""Multi-head attention layer.

    This module owns the query, key, value, and output projections. When
    ``key_value`` is omitted, queries, keys, and values are projected from the
    same input tensor. When ``key_value`` is passed, it is interpreted as
    unprojected context states from which keys and values are produced.

    Args:
        channels: The number of input and output channels.
        num_heads: The number of attention heads.
        qassmax: Whether to scale queries with :class:`QASSMax`.
        device: The device to use for module parameters.
        dtype: The dtype to use for module parameters.

    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        qassmax: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"`channels` ({channels}) must be divisible by `num_heads` "
                f"({num_heads})"
            )

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_heads = num_heads
        self.qkv_lin = Linear(channels, 3 * channels, **factory_kwargs)
        self.sdpa = SDPA(
            channels=channels // num_heads,
            num_heads=num_heads,
            qassmax=qassmax,
            **factory_kwargs,
        )
        self.out_lin = Linear(channels, channels, **factory_kwargs)

        torch.nn.init.zeros_(self.out_lin.weight)
        torch.nn.init.zeros_(self.out_lin.bias)

    def forward(
        self,
        query: Tensor,  # [..., Q, C]
        key_value: Tensor | None = None,  # [..., KV, C]
        seqused_key_value: Tensor | None = None,  # [...]
        attn_mask: Tensor | None = None,  # [..., Q, KV]
        rope: RotaryEmbedding | None = None,
    ) -> Tensor:  # [..., Q, C]
        r"""Forward pass of multi-head attention layer.

        Args:
            query: Unprojected query-side hidden states with shape
                ``[..., Q, C]``.
            key_value: Optional unprojected key/value-side hidden states with
                shape ``[..., KV, C]``. If omitted, ``query`` is used for
                self-attention.
            seqused_key_value: Optional valid key/value lengths with shape
                ``[...]`` and dtype ``torch.int32``.
            attn_mask: Optional boolean attention mask with shape
                ``[..., Q, KV]``. Entries set to ``True`` participate in
                attention.
            rope: Optional rotary positional embedding applied after
                projection.

        Returns:
            Tensor with shape ``[..., Q, C]``.

        """
        if key_value is None:
            query, key, value = self.qkv_lin(query).chunk(chunks=3, dim=-1)
        else:
            sections = [query.size(-1), 2 * query.size(-1)]
            q_weight, kv_weight = self.qkv_lin.weight.split(sections, dim=0)
            q_bias, kv_bias = self.qkv_lin.bias.split(sections, dim=0)
            query = F.linear(query, q_weight, q_bias)
            key, value = F.linear(key_value, kv_weight, kv_bias).chunk(2, -1)

        # [..., S, C] -> [..., S, H, C // H]
        query = query.unflatten(-1, [self.num_heads, -1])
        key = key.unflatten(-1, [self.num_heads, -1])
        value = value.unflatten(-1, [self.num_heads, -1])

        if rope is not None:
            query = rope(query)
            key = rope(key)
            assert query.dtype == key.dtype == value.dtype

        out = self.sdpa(
            query=query,  # [..., Q, H, C // H]
            key=key,  # [..., KV, H, C // H]
            value=value,  # [..., KV, H, C // H]
            seqused_key_value=seqused_key_value,  # [...]
            attn_mask=attn_mask,  # [..., Q, KV]
        )  # [..., Q, H, C // H]

        out = out.flatten(-2, -1)  # [..., Q, C]
        return self.out_lin(out)  # [..., Q, C]
