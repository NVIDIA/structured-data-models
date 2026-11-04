# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

    @overload
    def forward(
        self,
        query: Tensor,
        key_value: Tensor | KVCacheEntry | None = None,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        return_key_value: Literal[False] = False,
        batch_size_limit: int | Literal["auto"] | None = None,
        out: Tensor | None = None,
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
        batch_size_limit: int | Literal["auto"] | None = None,
        out: Tensor | None = None,
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
        batch_size_limit: int | Literal["auto"] | None = None,
        out: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, KVCacheEntry]: ...

    def forward(
        self,
        query: Tensor,  # [..., Q, C]
        key_value: Tensor | KVCacheEntry | None = None,  # [..., KV, C]
        seqused_key_value: Tensor | None = None,  # [...]
        attn_mask: Tensor | None = None,  # [..., KV]
        *,
        return_key_value: bool = False,
        batch_size_limit: int | Literal["auto"] | None = None,
        out: Tensor | None = None,
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
            out: The output tensor.

        Returns:
            Tensor with shape ``[..., Q, C]`` when ``return_key_value`` is
            ``False``. Otherwise, a tuple of the output tensor and a
            :class:`~sdm.cache.KVCacheEntry`.
        """
        if not isinstance(key_value, KVCacheEntry):
            if key_value is None:
                key_value = query
            if attn_mask is not None:
                attn_mask = attn_mask[..., None, None, :]  # [..., 1, 1, KV]
            key_value = self.inducing_block(
                query=self.inducing_points,  # [M, C]
                key_value=key_value,  # [..., KV, C]
                seqused_key_value=seqused_key_value,  # [...]
                attn_mask=attn_mask,  # [..., 1, 1, KV]
                batch_size_limit=batch_size_limit,
            )  # [..., M, C]
        return self.output_block(
            query=query,  # [..., Q, C]
            key_value=key_value,  # [..., M, C]
            return_key_value=return_key_value,
            batch_size_limit=batch_size_limit,
            out=out,
        )  # [..., Q, C]

    def summarize(
        self,
        key_value: Tensor,  # [..., KV, C]
        *,
        key_len: int,
        summary: tuple[Tensor, Tensor] | None = None,
        batch_size_limit: int | Literal["auto"] | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Accumulate the inducing points' attention over a key/value chunk.

        Calling this on consecutive chunks of a key/value sequence and passing
        the result to :meth:`induced_key_value` gives the same key/value
        projections of :math:`H` as attending to the whole sequence at once,
        up to rounding.

        Args:
            key_value: The key/value chunk with shape ``[..., KV, C]``.
            key_len: The length of the whole key/value sequence.
            summary: The summary of the preceding chunks, if any.
            batch_size_limit: Maximum number of batch elements processed at
                once.

        Returns:
            The summary of the chunks so far.
        """
        out, lse = self.inducing_block.partial(
            query=self.inducing_points,  # [M, C]
            key_value=key_value,  # [..., KV, C]
            key_len=key_len,
            batch_size_limit=batch_size_limit,
        )  # [..., M, H, C // H], [..., M, H]
        if summary is None:
            return out, lse
        summary_out, summary_lse = summary
        total = torch.logaddexp(summary_lse, lse)  # [..., M, H]
        out = out.float() * (lse - total).exp()[..., None]
        out += summary_out.float() * (summary_lse - total).exp()[..., None]
        return out, total

    def induced_key_value(
        self,
        summary: tuple[Tensor, Tensor],
    ) -> KVCacheEntry:  # [..., M, H, C // H]
        r"""The key/value projections of :math:`H` from a summary.

        The result can be passed as ``key_value`` to :meth:`forward`.

        Args:
            summary: The summary of a whole key/value sequence returned by
                :meth:`summarize`.
        """
        h = self.inducing_block.finish(
            query=self.inducing_points,  # [M, C]
            out=summary[0],  # [..., M, H, C // H]
        )  # [..., M, C]
        # An empty query only projects the key/value.
        _, key_value = self.output_block(
            query=h[..., :0, :],
            key_value=h,
            return_key_value=True,
        )
        return key_value
