"""Attention modules for structured tensor models."""

from collections.abc import Callable, Iterator
from itertools import product
from typing import Any, Literal, cast, overload

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GELU, LayerNorm, Linear, Sequential

from sdm.cache import KVCacheEntry
from sdm.nn import RotaryEmbedding, _cudnn_varlen

_BatchTile = tuple[slice, ...]


def _is_nested(*tensors: Tensor | None) -> bool:
    """Return whether any present tensor uses a nested layout."""
    return any(x is not None and x.is_nested for x in tensors)


def _batch_shape(x: Tensor, tail_dims: int) -> torch.Size:
    """Return the batch shape before ``tail_dims`` non-batch dimensions."""
    if tail_dims == 0:
        return x.shape
    return x.shape[:-tail_dims]


def _broadcast_batch_shape(
    *operands: tuple[Tensor | None, int],
) -> torch.Size:
    """Return the broadcast batch shape of ``(tensor, tail_dims)`` operands."""
    shapes = [
        _batch_shape(x, tail_dims)
        for x, tail_dims in operands
        if x is not None
    ]
    return torch.broadcast_shapes(*shapes) if shapes else torch.Size()


def _tile_sizes(batch_shape: torch.Size, limit: int) -> tuple[int, ...]:
    """Choose a rectangular tile whose number of batch elements is bounded."""
    sizes = [1] * len(batch_shape)
    remaining = limit
    for dim in range(len(batch_shape) - 1, -1, -1):
        sizes[dim] = min(batch_shape[dim], remaining)
        remaining = max(1, remaining // max(1, sizes[dim]))
    return tuple(sizes)


def _batch_tiles(batch_shape: torch.Size, limit: int) -> Iterator[_BatchTile]:
    """Yield rectangular slices with at most ``limit`` batch elements."""
    if batch_shape.numel() == 0:
        return
    if len(batch_shape) == 0:
        yield ()
        return

    sizes = _tile_sizes(batch_shape, limit)
    ranges = [range(0, size, step) for size, step in zip(batch_shape, sizes)]
    for starts in product(*ranges):
        yield tuple(
            slice(start, min(start + step, size))
            for start, step, size in zip(starts, sizes, batch_shape)
        )


def _slice_batch(
    x: Tensor,
    batch_shape: torch.Size,
    tile: _BatchTile,
    tail_dims: int,
) -> Tensor:
    """Slice one broadcast tile without expanding the complete batch.

    Missing leading batch dimensions and singleton batch dimensions remain
    singleton so the inner operation broadcasts only the current tile.
    """
    source_shape = _batch_shape(x, tail_dims)
    num_leading = len(batch_shape) - len(source_shape)
    if num_leading < 0:
        raise ValueError(
            f"Cannot broadcast batch shape {tuple(source_shape)} to "
            f"{tuple(batch_shape)}"
        )

    for _ in range(num_leading):
        x = x.unsqueeze(0)
    padded_shape = (1,) * num_leading + tuple(source_shape)
    index: list[slice] = []
    for source_size, output_size, output_slice in zip(
        padded_shape, batch_shape, tile
    ):
        if source_size == 1:
            index.append(slice(0, 1))
        elif source_size == output_size:
            index.append(output_slice)
        else:
            raise ValueError(
                f"Cannot broadcast batch shape {tuple(source_shape)} to "
                f"{tuple(batch_shape)}"
            )
    index.extend([slice(None)] * tail_dims)
    return x[tuple(index)]


def _copy_batch_tile(
    out: Tensor,
    value: Tensor,
    tile: _BatchTile,
    tail_dims: int,
) -> None:
    """Copy ``value`` into a batch tile of a preallocated output."""
    index = tile + (slice(None),) * tail_dims
    out[index].copy_(value)


def _attention_batch_shape(
    query: Tensor,
    key_value: Tensor | KVCacheEntry | None,
    seqused_key_value: Tensor | None,
    attn_mask: Tensor | None,
) -> torch.Size:
    """Return the output batch shape for an attention invocation."""
    operands: list[tuple[Tensor | None, int]] = [(query, 2)]
    if isinstance(key_value, KVCacheEntry):
        operands.extend([(key_value.key, 3), (key_value.value, 3)])
    elif isinstance(key_value, Tensor):
        operands.append((key_value, 2))
    else:
        operands.append((query, 2))
    operands.extend([(seqused_key_value, 0), (attn_mask, 2)])
    return _broadcast_batch_shape(*operands)


def _slice_attention_inputs(
    query: Tensor,
    key_value: Tensor | KVCacheEntry | None,
    seqused_key_value: Tensor | None,
    attn_mask: Tensor | None,
    batch_shape: torch.Size,
    tile: _BatchTile,
) -> tuple[
    Tensor,
    Tensor | KVCacheEntry | None,
    Tensor | None,
    Tensor | None,
]:
    """Slice all operands for one broadcast attention tile."""
    query = _slice_batch(query, batch_shape, tile, tail_dims=2)
    if isinstance(key_value, KVCacheEntry):
        key_value = KVCacheEntry(
            key=_slice_batch(key_value.key, batch_shape, tile, tail_dims=3),
            value=_slice_batch(
                key_value.value, batch_shape, tile, tail_dims=3
            ),
        )
    elif isinstance(key_value, Tensor):
        key_value = _slice_batch(key_value, batch_shape, tile, tail_dims=2)
    if seqused_key_value is not None:
        seqused_key_value = _slice_batch(
            seqused_key_value, batch_shape, tile, tail_dims=0
        )
    if attn_mask is not None:
        attn_mask = _slice_batch(attn_mask, batch_shape, tile, tail_dims=2)
    return query, key_value, seqused_key_value, attn_mask


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
    and extends it by arbitrary batch dimensions, :class:`QASSMax`-based
    temperature-scaling, and padding support for key/value pairs.

    Args:
        channels: The number of channels per attention head.
        num_query_heads: The number of query attention heads.
        num_key_value_heads: The number of key/value attention heads.
            Setting this below ``num_query_heads`` enables grouped-query
            attention (GQA); setting it to ``1`` enables multi-query attention
            (MQA). Must divide ``num_query_heads``. Defaults to
            ``num_query_heads`` (standard multi-head attention).
        qassmax: Whether to scale queries via :class:`QASSMax`.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        num_query_heads: int,
        num_key_value_heads: int | None = None,
        qassmax: bool = False,
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
            batch_size_limit: If set, run SDPA in rectangular broadcast-batch
                tiles containing at most this many elements. Only active when
                gradients are disabled and execution is not compiled. Nested
                tensor inputs currently execute unchunked.

        Returns:
            Tensor with shape ``[..., Q, Hq, C]``.
        """
        if batch_size_limit is not None and batch_size_limit <= 0:
            raise ValueError(
                f"`batch_size_limit` ({batch_size_limit}) must be positive"
            )
        # Preserve the established empty-query behavior without attempting to
        # broadcast it against non-empty key/value batch dimensions.
        if query.numel() == 0:
            return query
        if (
            batch_size_limit is not None
            and not _is_nested(query, key, value, attn_mask)
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
        ):
            chunked = self._chunked_forward(
                query,
                key,
                value,
                seqused_key_value,
                attn_mask,
                batch_size_limit,
            )
            if chunked is not None:
                return chunked

        return self._forward(query, key, value, seqused_key_value, attn_mask)

    def _forward(
        self,
        query: Tensor,  # [..., Q, Hq, C]
        key: Tensor,  # [..., KV, Hkv, C]
        value: Tensor,  # [..., KV, Hkv, C]
        seqused_key_value: Tensor | None,
        attn_mask: Tensor | None,
    ) -> Tensor:  # [..., Q, Hq, C]
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
            seqused_key_value = seqused_key_value.reshape(-1)
            if _cudnn_varlen.eligible(
                query,
                key,
                num_query_heads=self.num_query_heads,
                num_key_value_heads=self.num_key_value_heads,
            ):
                # cuDNN's native padding-mask support bounds attention to
                # the valid key/value region instead of masking it: 3.1-5.5x
                # over the boolean-mask kernels at D=64 ICL shapes and
                # ~2.0-2.4x at D=16 sites (see sdm/nn/_cudnn_varlen.py).
                out = _cudnn_varlen.cudnn_varlen_sdpa(
                    query.contiguous(),
                    key.contiguous(),
                    value.contiguous(),
                    seqused_key_value.contiguous(),
                )
                return out.view(batch_shape + out.size()[-3:])
            seqused_key_value = seqused_key_value.unsqueeze(-1)
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
        ).transpose(-3, -2)  # [B, Q, Hq, C]

        return out.view(batch_shape + out.size()[-3:])  # [..., Q, Hq, C]

    def _chunked_forward(
        self,
        query: Tensor,  # [..., Q, Hq, C]
        key: Tensor,  # [..., KV, Hkv, C]
        value: Tensor,  # [..., KV, Hkv, C]
        seqused_key_value: Tensor | None,
        attn_mask: Tensor | None,
        limit: int,
    ) -> Tensor | None:
        """Run SDPA over bounded rectangular broadcast-batch tiles."""
        batch_shape = _broadcast_batch_shape(
            (query, 3),
            (key, 3),
            (value, 3),
            (seqused_key_value, 0),
            (attn_mask, 2),
        )
        if batch_shape.numel() <= limit:
            return None

        out: Tensor | None = None
        for tile in _batch_tiles(batch_shape, limit):
            query_tile = _slice_batch(query, batch_shape, tile, tail_dims=3)
            key_tile = _slice_batch(key, batch_shape, tile, tail_dims=3)
            value_tile = _slice_batch(value, batch_shape, tile, tail_dims=3)
            seq_tile = (
                None
                if seqused_key_value is None
                else _slice_batch(
                    seqused_key_value, batch_shape, tile, tail_dims=0
                )
            )
            mask_tile = (
                None
                if attn_mask is None
                else _slice_batch(attn_mask, batch_shape, tile, tail_dims=2)
            )
            out_tile = self._forward(
                query_tile, key_tile, value_tile, seq_tile, mask_tile
            )
            if out is None:
                out = out_tile.new_empty((*batch_shape, *out_tile.shape[-3:]))
            _copy_batch_tile(out, out_tile, tile, tail_dims=3)

        assert out is not None
        return out


class Attention(torch.nn.Module):
    r"""Multi-head attention layer with grouped-query attention support.

    This module owns the query, key, value, and output projections.
    It performs self-attention when ``key_value`` is omitted and
    cross-attention when ``key_value`` is given.

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
    """

    def __init__(
        self,
        channels: int,
        num_query_heads: int,
        num_key_value_heads: int | None = None,
        qassmax: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
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
            **factory_kwargs,
        )
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
            seqused_key_value: Valid key/value lengths with shape ``[...]`` and
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
            attn_mask: Boolean attention mask with shape ``[..., Q, KV]``.
                Entries set to ``True`` participate in attention.
            rope: Rotary Positional Embedding applied after query/key
                projection.
            batch_size_limit: If set, run the full attention (query/key/value
                projections included) in chunks of at most this many
                broadcasted batch elements, writing into a preallocated
                output, to cap peak memory for very large batches. Only active
                in inference (gradients disabled) and eager (non-compiled)
                mode; nested tensor inputs execute unchunked. ``None`` disables
                it.
            return_key_value: Whether to return the computed key and value
                projections alongside the attention output.

        Returns:
            Tensor with shape ``[..., Q, C]`` when ``return_key_value`` is
            ``False``.
            Otherwise, a tuple of the output tensor and a
            :class:`~sdm.cache.KVCacheEntry`.
        """
        if batch_size_limit is not None and batch_size_limit <= 0:
            raise ValueError(
                f"`batch_size_limit` ({batch_size_limit}) must be positive"
            )
        if (
            batch_size_limit is not None
            and not _is_nested(
                query,
                key_value.key
                if isinstance(key_value, KVCacheEntry)
                else key_value,
                key_value.value
                if isinstance(key_value, KVCacheEntry)
                else None,
                attn_mask,
            )
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
        ):
            chunked = self._chunked_attend(
                query,
                key_value,
                seqused_key_value,
                attn_mask,
                rope,
                batch_size_limit,
                return_key_value,
            )
            if chunked is not None:
                return chunked

        out, key, value = self._attend(
            query, key_value, seqused_key_value, attn_mask, rope
        )
        if return_key_value:
            return out, KVCacheEntry(key=key, value=value)
        return out

    def _project_query(
        self,
        query: Tensor,
        rope: RotaryEmbedding | None,
    ) -> Tensor:
        """Project query channels and split them into attention heads."""
        q_weight = self.qkv_lin.weight[: self.q_dim]
        q_bias = self.qkv_lin.bias[: self.q_dim]
        query = F.linear(query, q_weight, q_bias)
        query = query.unflatten(-1, [self.num_query_heads, self.head_dim])
        return rope(query) if rope is not None else query

    def _project_key_value(
        self,
        key_value: Tensor,
        rope: RotaryEmbedding | None,
    ) -> KVCacheEntry:
        """Project compact native-shape key/value tensors for cache use."""
        kv_weight = self.qkv_lin.weight[self.q_dim :]
        kv_bias = self.qkv_lin.bias[self.q_dim :]
        key, value = F.linear(key_value, kv_weight, kv_bias).chunk(2, -1)
        key = key.unflatten(-1, [self.num_key_value_heads, self.head_dim])
        value = value.unflatten(-1, [self.num_key_value_heads, self.head_dim])
        if rope is not None:
            key = rope(key)
        return KVCacheEntry(key=key, value=value)

    def _build_cache(
        self,
        source: Tensor,
        rope: RotaryEmbedding | None,
        limit: int,
        source_transform: Callable[[Tensor], Tensor] | None = None,
    ) -> KVCacheEntry:
        """Project K/V in native-batch tiles and preserve compact shape."""
        batch_shape = source.shape[:-2]
        if batch_shape.numel() <= limit:
            if source_transform is not None:
                source = source_transform(source)
            return self._project_key_value(source, rope)

        key_out: Tensor | None = None
        value_out: Tensor | None = None
        for tile in _batch_tiles(batch_shape, limit):
            source_tile = _slice_batch(source, batch_shape, tile, tail_dims=2)
            if source_transform is not None:
                source_tile = source_transform(source_tile)
            projected = self._project_key_value(source_tile, rope)
            if key_out is None:
                key_out = projected.key.new_empty(
                    (*batch_shape, *projected.key.shape[-3:])
                )
                value_out = projected.value.new_empty(
                    (*batch_shape, *projected.value.shape[-3:])
                )
            _copy_batch_tile(key_out, projected.key, tile, tail_dims=3)
            assert value_out is not None
            _copy_batch_tile(value_out, projected.value, tile, tail_dims=3)

        assert key_out is not None
        assert value_out is not None
        return KVCacheEntry(key=key_out, value=value_out)

    def _attend(
        self,
        query: Tensor,  # [..., Q, C]
        key_value: Tensor | KVCacheEntry | None,  # [..., KV, C]
        seqused_key_value: Tensor | None,  # [...]
        attn_mask: Tensor | None,  # [..., Q, KV]
        rope: RotaryEmbedding | None,
    ) -> tuple[Tensor, Tensor, Tensor]:  # out [..., Q, C]; key/value heads
        if isinstance(key_value, KVCacheEntry):
            query = self._project_query(query, rope)
            if (
                key_value.key.dtype != query.dtype
                or key_value.value.dtype != query.dtype
            ):
                # Compare against the projected query so autocast runs (which
                # cast at the projection) validate correctly.
                raise ValueError(
                    f"Key/value projections were cached under dtypes "
                    f"'{key_value.key.dtype}'/'{key_value.value.dtype}' but "
                    f"the query projects to dtype '{query.dtype}'. Re-run "
                    f"the caching step under the current dtype "
                    f"configuration, or restore the configuration that was "
                    f"active when caching."
                )
            key = key_value.key
            value = key_value.value
        else:
            if key_value is None:
                query, key, value = self.qkv_lin(query).split(
                    [self.q_dim, self.kv_dim, self.kv_dim], dim=-1
                )
            else:
                sections = [self.q_dim, 2 * self.kv_dim]
                q_weight, kv_weight = self.qkv_lin.weight.split(
                    sections, dim=0
                )
                q_bias, kv_bias = self.qkv_lin.bias.split(sections, dim=0)
                query = F.linear(query, q_weight, q_bias)
                key, value = F.linear(key_value, kv_weight, kv_bias).chunk(
                    2, -1
                )

            # [..., S, C] -> [..., S, H, C // H].
            query = query.unflatten(-1, [self.num_query_heads, self.head_dim])
            key = key.unflatten(-1, [self.num_key_value_heads, self.head_dim])
            value = value.unflatten(
                -1, [self.num_key_value_heads, self.head_dim]
            )

            if rope is not None:
                query = rope(query)
                key = rope(key)

        if rope is not None:
            assert query.dtype == key.dtype == value.dtype

        out = self.sdpa(
            query=query,  # [..., Q, Hq, C // Hq]
            key=key,  # [..., KV, Hkv, C // Hq]
            value=value,  # [..., KV, Hkv, C // Hq]
            seqused_key_value=seqused_key_value,  # [...]
            attn_mask=attn_mask,  # [..., Q, KV]
        )  # [..., Q, Hq, C // Hq]

        out = out.flatten(-2, -1)  # [..., Q, C]
        out = self.out_lin(out)  # [..., Q, C]
        return out, key, value

    def _chunked_attend(
        self,
        query: Tensor,  # [..., Q, C]
        key_value: Tensor | KVCacheEntry | None,  # [..., KV, C]
        seqused_key_value: Tensor | None,  # [...]
        attn_mask: Tensor | None,  # [..., Q, KV]
        rope: RotaryEmbedding | None,
        limit: int,
        return_key_value: bool,
    ) -> Tensor | tuple[Tensor, KVCacheEntry] | None:
        r"""Run full attention over bounded rectangular broadcast tiles."""
        batch_shape = _attention_batch_shape(
            query, key_value, seqused_key_value, attn_mask
        )
        if batch_shape.numel() <= limit:
            return None

        if query.numel() == 0:
            if not return_key_value:
                return None
            if isinstance(key_value, KVCacheEntry):
                cache = key_value
            else:
                cache_source = query if key_value is None else key_value
                cache = self._build_cache(cache_source, rope, limit)
            out, _, _ = self._attend(
                query, cache, seqused_key_value, attn_mask, rope
            )
            return out, cache

        cache: KVCacheEntry | None = None
        tiled_key_value = key_value
        if return_key_value:
            if isinstance(key_value, KVCacheEntry):
                cache = key_value
            else:
                cache_source = query if key_value is None else key_value
                cache = self._build_cache(cache_source, rope, limit)
            tiled_key_value = cache

        out: Tensor | None = None
        for tile in _batch_tiles(batch_shape, limit):
            query_tile, kv_tile, seq_tile, mask_tile = _slice_attention_inputs(
                query,
                tiled_key_value,
                seqused_key_value,
                attn_mask,
                batch_shape,
                tile,
            )
            out_tile, _, _ = self._attend(
                query_tile, kv_tile, seq_tile, mask_tile, rope
            )
            if out is None:
                out = out_tile.new_empty((*batch_shape, *out_tile.shape[-2:]))
            _copy_batch_tile(out, out_tile, tile, tail_dims=2)

        assert out is not None
        if return_key_value:
            assert cache is not None
            return out, cache
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
        norm_bias: Whether :class:`~torch.nn.LayerNorm` uses a learnable bias.
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
        norm_bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.q_norm = LayerNorm(channels, bias=norm_bias, **factory_kwargs)
        self.kv_norm = LayerNorm(channels, bias=norm_bias, **factory_kwargs)
        self.attn = Attention(
            channels=channels,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            qassmax=qassmax,
            **factory_kwargs,
        )
        self.mlp = Sequential(
            LayerNorm(channels, bias=norm_bias, **factory_kwargs),
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
            batch_size_limit: If set, run the whole block (attention and the
                feed-forward MLP) in chunks of at most this many broadcasted
                batch elements, writing into a preallocated output, to cap
                peak memory for very large batches. Only active in inference
                (gradients disabled) and eager (non-compiled) mode; nested
                tensor inputs execute unchunked. ``None`` disables it.
            return_key_value: Whether to return the computed key and value
                projections alongside the block output.

        Returns:
            Tensor with shape ``[..., Q, C]`` when ``return_key_value`` is
            ``False``.
            Otherwise, a tuple of the output tensor and a
            :class:`~sdm.cache.KVCacheEntry`.
        """
        if batch_size_limit is not None and batch_size_limit <= 0:
            raise ValueError(
                f"`batch_size_limit` ({batch_size_limit}) must be positive"
            )
        if (
            batch_size_limit is not None
            and not _is_nested(
                query,
                key_value.key
                if isinstance(key_value, KVCacheEntry)
                else key_value,
                key_value.value
                if isinstance(key_value, KVCacheEntry)
                else None,
                attn_mask,
            )
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
        ):
            chunked = self._chunked_forward(
                query,
                key_value,
                seqused_key_value,
                attn_mask,
                rope,
                batch_size_limit,
                return_key_value,
            )
            if chunked is not None:
                return chunked

        if isinstance(key_value, Tensor):
            key_value = self.kv_norm(key_value)
        attn_result = self.attn(
            query=self.q_norm(query),
            key_value=key_value,
            seqused_key_value=seqused_key_value,
            attn_mask=attn_mask,
            rope=rope,
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

    def _block(
        self,
        query: Tensor,  # [..., Q, C]
        key_value: Tensor | KVCacheEntry | None,  # [..., KV, C]
        seqused_key_value: Tensor | None,  # [...]
        attn_mask: Tensor | None,  # [..., Q, KV]
        rope: RotaryEmbedding | None,
    ) -> Tensor:  # [..., Q, C]
        normed_kv = key_value
        if isinstance(normed_kv, Tensor):
            normed_kv = self.kv_norm(normed_kv)
        attn_out = self.attn(
            query=self.q_norm(query),
            key_value=normed_kv,
            seqused_key_value=seqused_key_value,
            attn_mask=attn_mask,
            rope=rope,
        )
        out = query + attn_out
        return out + self.mlp(out)

    def _chunked_forward(
        self,
        query: Tensor,  # [..., Q, C]
        key_value: Tensor | KVCacheEntry | None,  # [..., KV, C]
        seqused_key_value: Tensor | None,  # [...]
        attn_mask: Tensor | None,  # [..., Q, KV]
        rope: RotaryEmbedding | None,
        limit: int,
        return_key_value: bool,
    ) -> Tensor | tuple[Tensor, KVCacheEntry] | None:
        r"""Run the whole block over bounded rectangular broadcast tiles."""
        batch_shape = _attention_batch_shape(
            query, key_value, seqused_key_value, attn_mask
        )
        if batch_shape.numel() <= limit:
            return None

        if query.numel() == 0:
            if not return_key_value:
                return None
            if isinstance(key_value, KVCacheEntry):
                cache = key_value
            else:
                cache_source = query if key_value is None else key_value
                source_transform = (
                    self.q_norm if key_value is None else self.kv_norm
                )
                cache = self.attn._build_cache(
                    cache_source,
                    rope,
                    limit,
                    source_transform=source_transform,
                )
            return (
                self._block(
                    query,
                    cache,
                    seqused_key_value,
                    attn_mask,
                    rope,
                ),
                cache,
            )

        cache: KVCacheEntry | None = None
        tiled_key_value = key_value
        if return_key_value:
            if isinstance(key_value, KVCacheEntry):
                cache = key_value
            else:
                cache_source = query if key_value is None else key_value
                source_transform = (
                    self.q_norm if key_value is None else self.kv_norm
                )
                cache = self.attn._build_cache(
                    cache_source,
                    rope,
                    limit,
                    source_transform=source_transform,
                )
            tiled_key_value = cache

        out: Tensor | None = None
        for tile in _batch_tiles(batch_shape, limit):
            query_tile, kv_tile, seq_tile, mask_tile = _slice_attention_inputs(
                query,
                tiled_key_value,
                seqused_key_value,
                attn_mask,
                batch_shape,
                tile,
            )
            out_tile = self._block(
                query_tile, kv_tile, seq_tile, mask_tile, rope
            )
            if out is None:
                out = out_tile.new_empty((*batch_shape, *out_tile.shape[-2:]))
            _copy_batch_tile(out, out_tile, tile, tail_dims=2)

        assert out is not None
        if return_key_value:
            assert cache is not None
            return out, cache
        return out
