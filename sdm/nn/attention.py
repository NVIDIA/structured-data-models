"""Attention modules for structured tensor models."""

from collections.abc import Callable
from math import prod
from typing import Any, Literal, cast, overload

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GELU, Linear, Sequential

from sdm.cache import KVCacheEntry
from sdm.nn import RotaryEmbedding
from sdm.nn.resolver import normalization_resolver


def _resolve_batch_size_limit(batch_size_limit: int | None) -> int:
    if batch_size_limit is None:
        return 65_535
    return min(batch_size_limit, 65_535)


def _batch_chunk(
    tensor: Tensor,
    batch_shape: torch.Size,
    trailing_dims: int,
    start: int,
    end: int,
) -> Tensor:
    trailing_shape = tensor.size()[-trailing_dims:] if trailing_dims else ()
    tensor = tensor.expand(batch_shape + trailing_shape)
    if len(batch_shape) == 1:
        return tensor.narrow(0, start, end - start)

    flat_index = torch.arange(start, end, device=tensor.device)
    batch_indices: list[Tensor] = []
    for size in reversed(batch_shape):
        batch_indices.append(flat_index % size)
        flat_index = flat_index // size
    return tensor[tuple(reversed(batch_indices))]


def _optional_batch_chunk(
    tensor: Tensor | None,
    batch_shape: torch.Size,
    trailing_dims: int,
    start: int,
    end: int,
) -> Tensor | None:
    if tensor is None:
        return None
    return _batch_chunk(tensor, batch_shape, trailing_dims, start, end)


def _attention_batch_shape(
    query: Tensor,
    key_value: Tensor | KVCacheEntry | None,
    seqused_key_value: Tensor | None,
    attn_mask: Tensor | None,
) -> torch.Size:
    batch_shapes = [query.size()[:-2]]
    if isinstance(key_value, Tensor):
        batch_shapes.append(key_value.size()[:-2])
    elif isinstance(key_value, KVCacheEntry):
        batch_shapes.extend(
            [key_value.key.size()[:-3], key_value.value.size()[:-3]]
        )
    if seqused_key_value is not None:
        batch_shapes.append(seqused_key_value.size())
    if attn_mask is not None:
        batch_shapes.append(attn_mask.size()[:-2])
    return torch.broadcast_shapes(*batch_shapes)


def _chunk_key_value(
    key_value: Tensor | KVCacheEntry | None,
    batch_shape: torch.Size,
    start: int,
    end: int,
) -> Tensor | KVCacheEntry | None:
    if isinstance(key_value, Tensor):
        return _batch_chunk(key_value, batch_shape, 2, start, end)
    if isinstance(key_value, KVCacheEntry):
        return KVCacheEntry(
            key=_batch_chunk(key_value.key, batch_shape, 3, start, end),
            value=_batch_chunk(key_value.value, batch_shape, 3, start, end),
        )
    return None


def _chunk_attention(
    forward: Callable[..., object],
    query: Tensor,
    key_value: Tensor | KVCacheEntry | None,
    seqused_key_value: Tensor | None,
    attn_mask: Tensor | None,
    rope: RotaryEmbedding | None,
    return_key_value: bool,
    batch_size_limit: int,
) -> Tensor | tuple[Tensor, KVCacheEntry] | None:
    batch_shape = _attention_batch_shape(
        query=query,
        key_value=key_value,
        seqused_key_value=seqused_key_value,
        attn_mask=attn_mask,
    )
    batch_size = prod(batch_shape)
    if batch_size <= batch_size_limit:
        return None

    # SDPA returns an empty query before broadcasting its batch dimensions.
    # Preserve that behavior when another input would otherwise expand them.
    if query.size(-2) == 0 and query.size()[:-2] != batch_shape:
        return None

    if return_key_value:
        if isinstance(key_value, KVCacheEntry):
            return None
        cache_batch_shape = (
            query.size()[:-2] if key_value is None else key_value.size()[:-2]
        )
        if cache_batch_shape != batch_shape:
            return None

    query_size = query.size()[-2:]
    out: Tensor | None = None
    out_key: Tensor | None = None
    out_value: Tensor | None = None
    key_size: tuple[int, ...] | None = None
    value_size: tuple[int, ...] | None = None
    for start in range(0, batch_size, batch_size_limit):
        end = min(start + batch_size_limit, batch_size)
        chunk_result = forward(
            query=_batch_chunk(query, batch_shape, 2, start, end),
            key_value=_chunk_key_value(key_value, batch_shape, start, end),
            seqused_key_value=_optional_batch_chunk(
                seqused_key_value, batch_shape, 0, start, end
            ),
            attn_mask=_optional_batch_chunk(
                attn_mask, batch_shape, 2, start, end
            ),
            rope=rope,
            return_key_value=return_key_value,
            batch_size_limit=batch_size_limit,
        )
        if return_key_value:
            assert isinstance(chunk_result, tuple)
            chunk, chunk_key_value = chunk_result
            assert isinstance(chunk_key_value, KVCacheEntry)
        else:
            assert isinstance(chunk_result, Tensor)
            chunk = chunk_result
        assert isinstance(chunk, Tensor)
        if out is None:
            out = chunk.new_empty((batch_size, *query_size))
        out[start:end].copy_(chunk.reshape(end - start, *query_size))
        del chunk

        if return_key_value:
            key_size = chunk_key_value.key.size()[-3:]
            value_size = chunk_key_value.value.size()[-3:]
            if out_key is None:
                out_key = chunk_key_value.key.new_empty(
                    (batch_size, *key_size)
                )
                out_value = chunk_key_value.value.new_empty(
                    (batch_size, *value_size)
                )
            assert out_value is not None
            out_key[start:end].copy_(
                chunk_key_value.key.reshape(end - start, *key_size)
            )
            out_value[start:end].copy_(
                chunk_key_value.value.reshape(end - start, *value_size)
            )
            del chunk_key_value
        del chunk_result
    assert out is not None
    out = out.view(batch_shape + query_size)
    if not return_key_value:
        return out

    assert out_key is not None
    assert out_value is not None
    assert key_size is not None
    assert value_size is not None
    return out, KVCacheEntry(
        key=out_key.view(batch_shape + key_size),
        value=out_value.view(batch_shape + value_size),
    )


class QASSMax(torch.nn.Module):
    r"""Query-Aware Scalable SoftMax (QASSMax).

    This scaling method was introduced in the `"TabICLv2: A better, faster,
    scalable, and open tabular foundation model"
    <https://arxiv.org/abs/2602.11139>`_ paper as a temperature-scaling method
    for attention.

    For a query tensor :math:`q` and key length :math:`n`, this module returns
    a scaled query

    .. math::

        \tilde{q}_{hi} = q_{hi} \cdot
          \mathrm{MLP}_{\mathrm{base}}(\log n)_{hi} \cdot
          \left(1 + \tanh(\mathrm{MLP}_{\mathrm{gate}}(q_h)_i)\right),

    where :math:`h` indexes attention heads, :math:`i` indexes head channels,
    :math:`\mathrm{MLP}_{\mathrm{base}}` maps the log key length to per-head,
    per-channel scale factors, and :math:`\mathrm{MLP}_{\mathrm{gate}}` maps
    each per-head query vector to a bounded query-dependent gate.

    Multiplying the query scales the subsequent attention logits while keeping
    the attention computation compatible with standard softmax kernels.
    The length-dependent factor counteracts attention fading as the number of
    keys grows, while the query-dependent gate lets the scale vary across
    queries.

    Args:
        channels: The number of channels per attention head.
        num_heads: The number of query attention heads.
        hidden_channels: The hidden width of the scale and gate MLPs.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        hidden_channels: int = 64,
        device: torch.device | str | None = None,
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
        r"""The forward pass.

        Args:
            query: The query tensor to scale, with shape ``[..., S, H, C]``.
                ``S`` is the query sequence length, ``H`` is the number of
                attention heads, and ``C`` is the channels per head.
            key_len: The number of valid keys used to scale each query.
                An ``int`` denotes one shared key length for every query.
                A tensor shaped ``[..., 1]`` denotes one key length per batch
                item, shared across all query positions in ``S``.
                A tensor shaped ``[..., S]`` denotes one individual key length
                per query position.

        Returns:
            Tensor with shape ``[..., S, H, C]``.
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
    r"""Scaled Dot-Product Attention (SDPA).

    This module wraps :func:`torch.nn.functional.scaled_dot_product_attention`
    and extends it by arbitrary batch dimensions, optional inference-time
    batch chunking, :class:`QASSMax`-based temperature-scaling, and padding
    support for key/value pairs.

    Args:
        channels: The number of channels per attention head.
        num_query_heads: The number of query attention heads.
        num_key_value_heads: The number of key/value attention heads.
            Setting this below ``num_query_heads`` enables grouped-query
            attention (GQA); setting it to ``1`` enables multi-query attention
            (MQA). Must divide ``num_query_heads``. Defaults to
            ``num_query_heads`` (standard multi-head attention).
        qassmax: Whether to scale queries via :class:`QASSMax`.
        scale: Scaling factor passed to
            :func:`torch.nn.functional.scaled_dot_product_attention`.
            ``None`` uses the default value of ``1 / sqrt(channels)``.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        num_query_heads: int,
        num_key_value_heads: int | None = None,
        qassmax: bool = False,
        scale: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_key_value_heads is None:
            num_key_value_heads = num_query_heads
        if num_query_heads % num_key_value_heads != 0:
            raise ValueError(
                f"`num_query_heads` ({num_query_heads}) must be divisible by "
                f"`num_key_value_heads` ({num_key_value_heads})"
            )

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.scale = scale
        self.qassmax: QASSMax | None = None
        if qassmax:
            self.qassmax = QASSMax(
                channels=channels,
                num_heads=num_query_heads,
                **factory_kwargs,
            )

    def forward(
        self,
        query: Tensor,  # [..., Q, Hq, C]
        key: Tensor,  # [..., KV, Hkv, C]
        value: Tensor,  # [..., KV, Hkv, C]
        seqused_key_value: Tensor | None = None,  # [...]
        attn_mask: Tensor | None = None,  # [..., Q, KV]
        *,
        batch_size_limit: int | None = None,
    ) -> Tensor:  # [..., Q, Hq, C]
        r"""The forward pass.

        Args:
            query: The query tensor with shape ``[..., Q, Hq, C]``.
                ``Q`` is the query sequence length, ``Hq`` is the number of
                query attention heads (``num_query_heads``), and ``C`` is the
                channels per head.
            key: The key tensor with shape ``[..., KV, Hkv, C]``.
                ``KV`` is the key/value sequence length and ``Hkv`` is the
                number of key/value heads (``num_key_value_heads``).
            value: The value tensor with shape ``[..., KV, Hkv, C]``.
            seqused_key_value: Valid key/value lengths with shape ``[...]`` and
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
            attn_mask: Boolean attention mask with shape ``[..., Q, KV]``.
                Entries set to ``True`` participate in attention.
            batch_size_limit: Maximum number of batch elements processed at
                once.

        Returns:
            Tensor with shape ``[..., Q, Hq, C]``.
        """
        batch_size_limit = _resolve_batch_size_limit(batch_size_limit)

        if query.numel() == 0:
            return query

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

        batch_shapes = [query.size()[:-3], key.size()[:-3], value.size()[:-3]]
        if seqused_key_value is not None:
            batch_shapes.append(seqused_key_value.size())
        if attn_mask is not None:
            batch_shapes.append(attn_mask.size()[:-2])
        batch_shape = torch.broadcast_shapes(*batch_shapes)

        if not self.training and not torch.compiler.is_compiling():
            batch_size = prod(batch_shape)
            if batch_size > batch_size_limit:
                query_size = query.size()[-3:]
                out: Tensor | None = None
                for start in range(0, batch_size, batch_size_limit):
                    end = min(start + batch_size_limit, batch_size)
                    chunk = self.forward(
                        query=_batch_chunk(query, batch_shape, 3, start, end),
                        key=_batch_chunk(key, batch_shape, 3, start, end),
                        value=_batch_chunk(value, batch_shape, 3, start, end),
                        seqused_key_value=_optional_batch_chunk(
                            seqused_key_value, batch_shape, 0, start, end
                        ),
                        attn_mask=_optional_batch_chunk(
                            attn_mask, batch_shape, 2, start, end
                        ),
                        batch_size_limit=batch_size_limit,
                    )
                    if out is None:
                        out = chunk.new_empty((batch_size, *query_size))
                    out[start:end].copy_(
                        chunk.reshape(end - start, *query_size)
                    )
                    del chunk
                assert out is not None
                return out.view(batch_shape + query_size)

        if self.qassmax is not None:
            if seqused_key_value is not None:
                key_len = seqused_key_value.unsqueeze(-1)
            elif attn_mask is not None and attn_mask.size(-1) > 1:
                key_len = attn_mask.sum(dim=-1)
            else:
                key_len = key.size(-3)
            query = self.qassmax(query, key_len=key_len)

        # Broadcast and flatten batch dimensions => [B, S, H, C].
        query_size = query.size()[-3:]
        key_size = key.size()[-3:]
        value_size = value.size()[-3:]

        if key_size[0] == 0:  # No key/value pairs - abort early:
            return query.new_zeros(batch_shape + query_size)

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
            query=query.transpose(-3, -2),  # [B, Hq, Q, C],
            key=key.transpose(-3, -2),  # [B, Hkv, KV, C],
            value=value.transpose(-3, -2),  # [B, Hkv, KV, C],
            attn_mask=attn_mask.unsqueeze(-3)  # [B, 1, Q, KV]
            if attn_mask is not None
            else None,
            enable_gqa=self.num_query_heads != self.num_key_value_heads,
            scale=self.scale,
        ).transpose(-3, -2)  # [B, Q, Hq, C]

        return out.view(batch_shape + out.size()[-3:])  # [..., Q, Hq, C]


class Attention(torch.nn.Module):
    r"""Multi-head attention layer with grouped-query attention support.

    This module owns the query, key, value, and output projections.
    It performs self-attention when ``key_value`` is omitted and
    cross-attention when ``key_value`` is given.
    Supplied transformation modules are registered as-is; ``device`` and
    ``dtype`` apply only to modules constructed by this class.
    Query and key transformations must preserve the headed tensor shape and
    dtype.

    Args:
        channels: The number of input and output channels.
        num_query_heads: The number of query attention heads.
            ``channels`` must be divisible by ``num_query_heads``.
        num_key_value_heads: The number of key/value attention heads.
            Setting this below ``num_query_heads`` enables grouped-query
            attention (GQA); setting it to ``1`` enables multi-query attention
            (MQA). Must divide ``num_query_heads``. Defaults to
            ``num_query_heads`` (standard multi-head attention).
        qassmax: Whether to scale queries with :class:`QASSMax`.
        device: The device.
        dtype: The dtype.
        query_transform: Transformation applied to projected query heads after
            rotary embedding and before scaled dot-product attention.
        key_transform: Transformation applied to newly projected key heads
            after rotary embedding and before scaled dot-product attention.
            Cached keys are already transformed and are not transformed again.
        scale: Scaling factor passed to
            :func:`torch.nn.functional.scaled_dot_product_attention`.
            ``None`` uses ``1 / sqrt(channels_per_head)``.
    """

    def __init__(
        self,
        channels: int,
        num_query_heads: int,
        num_key_value_heads: int | None = None,
        qassmax: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        query_transform: torch.nn.Module | None = None,
        key_transform: torch.nn.Module | None = None,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if num_key_value_heads is None:
            num_key_value_heads = num_query_heads
        if channels % num_query_heads != 0:
            raise ValueError(
                f"`channels` ({channels}) must be divisible by "
                f"`num_query_heads` ({num_query_heads})"
            )

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = channels // num_query_heads
        # Query projection spans all channels; key/value span fewer heads.
        self.q_dim = num_query_heads * self.head_dim  # == channels
        self.kv_dim = num_key_value_heads * self.head_dim

        self.qkv_lin = Linear(
            channels, self.q_dim + 2 * self.kv_dim, **factory_kwargs
        )
        self.sdpa = SDPA(
            channels=self.head_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            qassmax=qassmax,
            scale=scale,
            **factory_kwargs,
        )
        self.query_transform = query_transform
        self.key_transform = key_transform
        self.out_lin = Linear(channels, channels, **factory_kwargs)

        torch.nn.init.zeros_(self.out_lin.weight)
        torch.nn.init.zeros_(self.out_lin.bias)

    @overload
    def forward(
        self,
        query: Tensor,
        key_value: Tensor | KVCacheEntry | None = None,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        rope: RotaryEmbedding | None = None,
        *,
        return_key_value: Literal[False] = False,
        batch_size_limit: int | None = None,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        query: Tensor,
        key_value: Tensor | KVCacheEntry | None = None,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        rope: RotaryEmbedding | None = None,
        *,
        return_key_value: Literal[True],
        batch_size_limit: int | None = None,
    ) -> tuple[Tensor, KVCacheEntry]: ...

    @overload
    def forward(
        self,
        query: Tensor,
        key_value: Tensor | KVCacheEntry | None = None,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        rope: RotaryEmbedding | None = None,
        *,
        return_key_value: bool,
        batch_size_limit: int | None = None,
    ) -> Tensor | tuple[Tensor, KVCacheEntry]: ...

    def forward(
        self,
        query: Tensor,  # [..., Q, C]
        key_value: Tensor | KVCacheEntry | None = None,  # [..., KV, C]
        seqused_key_value: Tensor | None = None,  # [...]
        attn_mask: Tensor | None = None,  # [..., Q, KV]
        rope: RotaryEmbedding | None = None,
        return_key_value: bool = False,
        *,
        batch_size_limit: int | None = None,
    ) -> Tensor | tuple[Tensor, KVCacheEntry]:  # [..., Q, C]
        r"""The forward pass.

        Args:
            query: The query tensor with shape ``[..., Q, C]``.
                ``Q`` is the query sequence length, ``C`` is the number of
                channels.
            key_value: The key/value tensor with shape ``[..., KV, C]`` or
                precomputed key/value projections as a
                :class:`~sdm.cache.KVCacheEntry`.
                ``KV`` is the key/value sequence length.
                If omitted, ``query`` is used for self-attention.
                Cached keys have already received rotary embedding and the
                configured key transformation.
            seqused_key_value: Valid key/value lengths with shape ``[...]`` and
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
            attn_mask: Boolean attention mask with shape ``[..., Q, KV]``.
                Entries set to ``True`` participate in attention.
            rope: Rotary Positional Embedding applied after query/key
                projection.
            return_key_value: Whether to return the computed key and value
                projections alongside the attention output.
            batch_size_limit: Maximum number of batch elements processed at
                once.

        Returns:
            Tensor with shape ``[..., Q, C]`` when ``return_key_value`` is
            ``False``.
            Otherwise, a tuple of the output tensor and a
            :class:`~sdm.cache.KVCacheEntry`.
        """
        batch_size_limit = _resolve_batch_size_limit(batch_size_limit)
        if not self.training and not torch.compiler.is_compiling():
            chunked_result = _chunk_attention(
                forward=self.forward,
                query=query,
                key_value=key_value,
                seqused_key_value=seqused_key_value,
                attn_mask=attn_mask,
                rope=rope,
                return_key_value=return_key_value,
                batch_size_limit=batch_size_limit,
            )
            if chunked_result is not None:
                return chunked_result

        if isinstance(key_value, KVCacheEntry):
            q_weight = self.qkv_lin.weight[: self.q_dim]
            q_bias = self.qkv_lin.bias[: self.q_dim]
            query = F.linear(query, q_weight, q_bias)
            if (
                key_value.key.dtype != query.dtype
                or key_value.value.dtype != query.dtype
            ):
                raise ValueError(
                    f"Key/value projections were cached under dtypes "
                    f"'{key_value.key.dtype}'/'{key_value.value.dtype}' but "
                    f"the query has dtype '{query.dtype}'"
                )
            key = key_value.key
            value = key_value.value
        elif key_value is None:
            query, key, value = self.qkv_lin(query).split(
                [self.q_dim, self.kv_dim, self.kv_dim], dim=-1
            )
        else:
            sections = [self.q_dim, 2 * self.kv_dim]
            q_weight, kv_weight = self.qkv_lin.weight.split(sections, dim=0)
            q_bias, kv_bias = self.qkv_lin.bias.split(sections, dim=0)
            query = F.linear(query, q_weight, q_bias)
            key, value = F.linear(key_value, kv_weight, kv_bias).chunk(2, -1)

        # [..., S, C] -> [..., S, H, C // H], with separate query/kv heads.
        query = query.unflatten(-1, [self.num_query_heads, self.head_dim])
        if not isinstance(key_value, KVCacheEntry):
            key = key.unflatten(-1, [self.num_key_value_heads, self.head_dim])
            value = value.unflatten(
                -1, [self.num_key_value_heads, self.head_dim]
            )

        if rope is not None:
            query = rope(query)
            if not isinstance(key_value, KVCacheEntry):
                key = rope(key)
            assert query.dtype == key.dtype == value.dtype

        if self.query_transform is not None:
            query = self.query_transform(query)
        if self.key_transform is not None and not isinstance(
            key_value, KVCacheEntry
        ):
            key = self.key_transform(key)

        out = self.sdpa(
            query=query,  # [..., Q, Hq, C // Hq]
            key=key,  # [..., KV, Hkv, C // Hq]
            value=value,  # [..., KV, Hkv, C // Hq]
            seqused_key_value=seqused_key_value,  # [...]
            attn_mask=attn_mask,  # [..., Q, KV]
            batch_size_limit=batch_size_limit,
        )  # [..., Q, Hq, C // Hq]

        out = out.flatten(-2, -1)  # [..., Q, C]
        out = self.out_lin(out)  # [..., Q, C]
        if return_key_value:
            return out, KVCacheEntry(key=key, value=value)
        return out


class TransformerBlock(torch.nn.Module):
    r"""Transformer block with pre-norm attention and feedforward modules.

    Supports optional :class:`RotaryEmbedding` on projected query/key tensors
    and optional :class:`QASSMax` query scaling inside attention.

    Args:
        channels: The number of input and output channels.
        num_query_heads: The number of query attention heads.
        feedforward_channels: The hidden width of the MLP.
        num_key_value_heads: The number of key/value attention heads.
            Defaults to ``num_query_heads`` (standard multi-head attention).
        qassmax: Whether to scale queries with :class:`QASSMax`.
        norm: The normalization layer name or a callable returning the
            normalization layer. The callable is invoked once per norm site,
            so each of the three sites gets a fresh instance. A module
            instance is shared across all three sites.
        norm_kwargs: Additional keyword arguments passed to the normalization
            layer constructor. Takes precedence over ``device`` and
            ``dtype``.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        num_query_heads: int,
        feedforward_channels: int,
        num_key_value_heads: int | None = None,
        qassmax: bool = False,
        norm: str | Callable[..., torch.nn.Module] = "layer_norm",
        norm_kwargs: dict[str, Any] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        # User `norm_kwargs` win; `device`/`dtype` fill unspecified keys.
        norm_kwargs = {**factory_kwargs, **(norm_kwargs or {})}

        self.q_norm = normalization_resolver(norm, channels, **norm_kwargs)
        self.kv_norm = normalization_resolver(norm, channels, **norm_kwargs)
        self.attn = Attention(
            channels=channels,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            qassmax=qassmax,
            **factory_kwargs,
        )
        self.mlp = Sequential(
            normalization_resolver(norm, channels, **norm_kwargs),
            Linear(channels, feedforward_channels, **factory_kwargs),
            GELU(),
            Linear(feedforward_channels, channels, **factory_kwargs),
        )

        torch.nn.init.zeros_(cast(Linear, self.mlp[-1]).weight)
        torch.nn.init.zeros_(cast(Linear, self.mlp[-1]).bias)

    @overload
    def forward(
        self,
        query: Tensor,
        key_value: Tensor | KVCacheEntry | None = None,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        rope: RotaryEmbedding | None = None,
        *,
        return_key_value: Literal[False] = False,
        batch_size_limit: int | None = None,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        query: Tensor,
        key_value: Tensor | KVCacheEntry | None = None,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        rope: RotaryEmbedding | None = None,
        *,
        return_key_value: Literal[True],
        batch_size_limit: int | None = None,
    ) -> tuple[Tensor, KVCacheEntry]: ...

    @overload
    def forward(
        self,
        query: Tensor,
        key_value: Tensor | KVCacheEntry | None = None,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        rope: RotaryEmbedding | None = None,
        *,
        return_key_value: bool,
        batch_size_limit: int | None = None,
    ) -> Tensor | tuple[Tensor, KVCacheEntry]: ...

    def forward(
        self,
        query: Tensor,  # [..., Q, C]
        key_value: Tensor | KVCacheEntry | None = None,  # [..., KV, C]
        seqused_key_value: Tensor | None = None,  # [...]
        attn_mask: Tensor | None = None,  # [..., Q, KV]
        rope: RotaryEmbedding | None = None,
        return_key_value: bool = False,
        *,
        batch_size_limit: int | None = None,
    ) -> Tensor | tuple[Tensor, KVCacheEntry]:  # [..., Q, C]
        r"""The forward pass.

        Args:
            query: The query tensor with shape ``[..., Q, C]``.
                ``Q`` is the query sequence length, ``C`` is the number of
                channels.
            key_value: The key/value tensor with shape ``[..., KV, C]`` or
                precomputed key/value projections as a
                :class:`~sdm.cache.KVCacheEntry`.
                ``KV`` is the key/value sequence length.
                If omitted, ``query`` is used for self-attention.
            seqused_key_value: Valid key/value lengths with shape ``[...]`` and
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
            attn_mask: Boolean attention mask with shape ``[..., Q, KV]``.
                Entries set to ``True`` participate in attention.
            rope: Rotary Positional Embedding applied after query/key
                projection.
            return_key_value: Whether to return the computed key and value
                projections alongside the block output.
            batch_size_limit: Maximum number of batch elements processed at
                once.

        Returns:
            Tensor with shape ``[..., Q, C]`` when ``return_key_value`` is
            ``False``.
            Otherwise, a tuple of the output tensor and a
            :class:`~sdm.cache.KVCacheEntry`.
        """
        batch_size_limit = _resolve_batch_size_limit(batch_size_limit)
        if not self.training and not torch.compiler.is_compiling():
            chunked_result = _chunk_attention(
                forward=self.forward,
                query=query,
                key_value=key_value,
                seqused_key_value=seqused_key_value,
                attn_mask=attn_mask,
                rope=rope,
                return_key_value=return_key_value,
                batch_size_limit=batch_size_limit,
            )
            if chunked_result is not None:
                return chunked_result

        if isinstance(key_value, Tensor):
            key_value = self.kv_norm(key_value)
        attn_result = self.attn(
            query=self.q_norm(query),
            key_value=key_value,
            seqused_key_value=seqused_key_value,
            attn_mask=attn_mask,
            rope=rope,
            batch_size_limit=batch_size_limit,
            return_key_value=return_key_value,
        )
        if return_key_value:
            attn_out, kv = attn_result
        else:
            attn_out = attn_result

        out = query + attn_out
        out = out + self.mlp(out)
        if return_key_value:
            return out, kv
        return out
