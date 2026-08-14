r"""Set-transformer modules for structured tensor models."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal, overload

import torch
from torch import Tensor
from torch.nn import Parameter

from sdm.cache import KVCacheEntry
from sdm.nn import QueryScaling, TransformerBlock


class InducedTransformerBlock(torch.nn.Module):
    r"""Transformer block using learned inducing points.

    Introduced in the `"Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper, the block routes attention
    through a small set of :math:`M` learned inducing points, reducing the cost
    of attending a query of size :math:`Q` to a key/value context of size
    :math:`K` from :math:`O(Q \cdot K)` to :math:`O((Q + K) \cdot M)`:

    .. math::

        H = \mathrm{TransformerBlock}_1(I, \mathrm{key\_value}), \quad
        \mathrm{out} = \mathrm{TransformerBlock}_2(\mathrm{query}, H),

    where :math:`I` are the learned inducing points and :math:`H` are the
    inducing points after attending to the key/value elements. Passing
    ``key_value=None`` recovers the induced self-attention block (ISAB).

    Args:
        channels: The number of input and output channels.
        num_query_heads: The number of query attention heads.
        feedforward_channels: The hidden width of the MLP.
        num_key_value_heads: The number of key/value attention heads.
            Defaults to ``num_query_heads`` (standard multi-head attention).
        num_inducing_points: The number of learned inducing points :math:`M`.
        norm: The normalization layer name or a callable returning the
            normalization layer. The callable is invoked once per norm site,
            so each site gets a fresh instance. A module instance is shared
            across all norm sites.
        norm_kwargs: Additional keyword arguments passed to the normalization
            layer constructor. Takes precedence over ``device`` and
            ``dtype``.
        query_scaling: Query scaling module to scale projected query heads
            before scaled dot-product attention, *e.g.*, :class:`QASSMax`.
        device: The device.
        dtype: The dtype.
        query_transform: Transformation applied to projected query heads in
            both internal transformer blocks. It is copied for each block.
        key_transform: Transformation applied to projected key heads in both
            internal transformer blocks. It is copied for each block.
        scale: Scaling factor passed to scaled dot-product attention in both
            internal transformer blocks.
        shared_attention_norm: Whether both internal transformer blocks use one
            ``norm`` instance for their query and key/value inputs.
        post_attention_norm: Whether both internal transformer blocks apply
            ``norm`` to attention outputs before their residual additions.
        feedforward_layer: Optional feed-forward layer constructor used by both
            internal transformer blocks. Each block constructs an independent
            instance. ``None`` uses the standard normalized GELU MLP.
        feedforward_kwargs: Additional keyword arguments passed to
            ``feedforward_layer`` by both internal transformer blocks.
    """

    def __init__(
        self,
        channels: int,
        num_query_heads: int,
        feedforward_channels: int,
        num_key_value_heads: int | None = None,
        num_inducing_points: int = 16,
        norm: str | Callable[..., torch.nn.Module] = "layer_norm",
        norm_kwargs: dict[str, Any] | None = None,
        query_scaling: QueryScaling | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        query_transform: torch.nn.Module | None = None,
        key_transform: torch.nn.Module | None = None,
        scale: float | None = None,
        shared_attention_norm: bool = False,
        post_attention_norm: bool = False,
        feedforward_layer: Callable[..., torch.nn.Module] | None = None,
        feedforward_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.transformer_1 = TransformerBlock(
            channels=channels,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            feedforward_channels=feedforward_channels,
            norm=norm,
            norm_kwargs=norm_kwargs,
            query_transform=deepcopy(query_transform),
            key_transform=deepcopy(key_transform),
            query_scaling=query_scaling,
            scale=scale,
            shared_attention_norm=shared_attention_norm,
            post_attention_norm=post_attention_norm,
            feedforward_layer=feedforward_layer,
            feedforward_kwargs=feedforward_kwargs,
            **factory_kwargs,
        )
        self.transformer_2 = TransformerBlock(
            channels=channels,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            feedforward_channels=feedforward_channels,
            norm=norm,
            norm_kwargs=norm_kwargs,
            query_transform=deepcopy(query_transform),
            key_transform=deepcopy(key_transform),
            scale=scale,
            shared_attention_norm=shared_attention_norm,
            post_attention_norm=post_attention_norm,
            feedforward_layer=feedforward_layer,
            feedforward_kwargs=feedforward_kwargs,
            **factory_kwargs,
        )

        self.inducing_points = Parameter(
            torch.empty(num_inducing_points, channels, **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.inducing_points, std=0.02)

    @overload
    def forward(
        self,
        query: Tensor,
        key_value: Tensor | KVCacheEntry | None = None,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
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
        *,
        return_key_value: bool,
        batch_size_limit: int | None = None,
    ) -> Tensor | tuple[Tensor, KVCacheEntry]: ...

    def forward(
        self,
        query: Tensor,  # [..., Q, C]
        key_value: Tensor | KVCacheEntry | None = None,  # [..., KV, C]
        seqused_key_value: Tensor | None = None,  # [...]
        attn_mask: Tensor | None = None,  # [..., KV]
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
                If omitted, ``query`` is used for induced self-attention.
            seqused_key_value: Valid key/value lengths with shape ``[...]`` and
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
            attn_mask: Boolean attention mask with shape ``[..., KV]``.
                Entries set to ``True`` participate in attention.
            return_key_value: Whether to return the computed key and value
                projections for the final attention site alongside the output.
            batch_size_limit: Maximum number of batch elements processed at
                once.

        Returns:
            Tensor with shape ``[..., Q, C]`` when ``return_key_value`` is
            ``False``. Otherwise, a tuple of the output tensor and a
            :class:`~sdm.cache.KVCacheEntry`.
        """
        if not isinstance(key_value, KVCacheEntry):
            if key_value is None:
                key_value = query
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(-2)  # [..., 1, KV]
            key_value = self.transformer_1(
                query=self.inducing_points,  # [M, C]
                key_value=key_value,  # [..., KV, C]
                seqused_key_value=seqused_key_value,  # [...]
                attn_mask=attn_mask,  # [..., 1, KV]
                batch_size_limit=batch_size_limit,
            )  # [..., M, C]
        return self.transformer_2(
            query=query,  # [..., Q, C]
            key_value=key_value,  # [..., M, C]
            return_key_value=return_key_value,
            batch_size_limit=batch_size_limit,
        )  # [..., Q, C]
