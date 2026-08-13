r"""Set-transformer modules for structured tensor models."""

from __future__ import annotations

from typing import Any, Literal, overload

import torch
from torch import Tensor
from torch.nn import Parameter

from sdm.cache import KVCacheEntry
from sdm.nn import TransformerBlock


class InducedTransformerBlock(torch.nn.Module):
    r"""Transformer block using learned inducing points.

    Introduced in the `"Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper, the block routes attention
    through a small set of :math:`M` learned inducing points, reducing the cost
    of attending a query of size :math:`Q` to a key/value context of size
    :math:`K` from :math:`O(Q \cdot K)` to :math:`O((Q + K) \cdot M)`:

    .. math::

        H = \mathrm{InducingBlock}_1(I, \mathrm{key\_value}), \quad
        \mathrm{out} = \mathrm{OutputBlock}_2(\mathrm{query}, H),

    where :math:`I` are the learned inducing points and :math:`H` are the
    inducing points after attending to the key/value elements. Passing
    ``key_value=None`` recovers the induced self-attention block (ISAB).

    Args:
        channels: The number of input and output channels.
        num_inducing_points: The number of learned inducing points :math:`M`.
        inducing_block: The :class:`TransformerBlock` that updates the learned
            inducing points from key/value context.
        output_block: The :class:`TransformerBlock` that updates the input
            queries from the induced context.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        num_inducing_points: int,
        inducing_block: TransformerBlock,
        output_block: TransformerBlock,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.inducing_block = inducing_block
        self.output_block = output_block

        self.inducing_points = Parameter(
            torch.empty(num_inducing_points, channels, **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.inducing_points, std=0.02)

    def induced_key_value(
        self,
        key_value: Tensor,
        *,
        batch_size_limit: int | None = None,
    ) -> KVCacheEntry:
        r"""Return the final attention's key/value projections.

        This summarizes ``key_value`` with the inducing points without
        materializing an output for every query element. The returned entry
        can be replayed by passing it as ``key_value`` to :meth:`forward`.

        Args:
            key_value: Input context summarized by the inducing points.
            batch_size_limit: Maximum number of batch elements processed at
                once.

        Returns:
            Projected keys and values for the block's final attention site.
        """
        inducing = self.inducing_block(
            query=self.inducing_points,
            key_value=key_value,
            batch_size_limit=batch_size_limit,
        )
        # The final attention's key/value projection is query-independent.
        # Use one inducing point as a disposable query instead of allocating
        # an output for the full context.
        _, cache = self.output_block(
            query=inducing[..., :1, :],
            key_value=inducing,
            return_key_value=True,
            batch_size_limit=batch_size_limit,
        )
        return cache

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
            key_value = self.inducing_block(
                query=self.inducing_points,  # [M, C]
                key_value=key_value,  # [..., KV, C]
                seqused_key_value=seqused_key_value,  # [...]
                attn_mask=attn_mask,  # [..., 1, KV]
                batch_size_limit=batch_size_limit,
            )  # [..., M, C]
        return self.output_block(
            query=query,  # [..., Q, C]
            key_value=key_value,  # [..., M, C]
            return_key_value=return_key_value,
            batch_size_limit=batch_size_limit,
        )  # [..., Q, C]
