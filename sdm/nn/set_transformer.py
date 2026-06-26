r"""Set-transformer modules for structured tensor models."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import ModuleList, Parameter

from sdm.nn.attention import TransformerBlock


class InducedSelfAttentionBlock(torch.nn.Module):
    r"""Induced self-attention block (ISAB) from the Set Transformer.

    The block reduces the cost of self-attention over a set of ``S`` elements
    from :math:`O(S^2)` to :math:`O(S)` by routing information through a small
    set of ``M`` learned inducing points:

    .. math::

        H = \mathrm{TransformerBlock}_1(I, X), \quad
        \mathrm{ISAB}(X) = \mathrm{TransformerBlock}_2(X, H),

    where :math:`X` are the input elements, :math:`I` are the learned inducing
    points, and :math:`H` are the inducing points after attending to ``X``.

    Introduced in `"Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_.

    Args:
        channels: Input and output channel count.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of the MLP in each block.
        num_inducing_points: Number of learned inducing points ``M``.
        qassmax: Whether to use :class:`~sdm.nn.QASSMax`. Applied only to the
            first attention, where the inducing points aggregate information
            from the variable-length context.
        device: The device to use for module parameters.
        dtype: The dtype to use for module parameters.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        feedforward_channels: int,
        num_inducing_points: int = 16,
        qassmax: bool = False,
        norm_bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_inducing_points = num_inducing_points

        self.transformer_1 = TransformerBlock(
            channels=channels,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            qassmax=qassmax,
            norm_bias=norm_bias,
            **factory_kwargs,
        )
        self.transformer_2 = TransformerBlock(
            channels=channels,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            norm_bias=norm_bias,
            **factory_kwargs,
        )

        self.inducing_points = Parameter(
            torch.empty(num_inducing_points, channels, **factory_kwargs)
        )

        torch.nn.init.trunc_normal_(self.inducing_points, std=0.02)

    def forward(
        self,
        x: Tensor,  # [..., S, C]
        context_size: int | None = None,
        context_mask: Tensor | None = None,  # [S]
        seqused_key_value: Tensor | None = None,  # [...]
    ) -> Tensor:  # [..., S, C]
        r"""Forward pass of the induced self-attention block.

        Args:
            x: Input set elements with shape ``[..., S, C]``, where ``S`` is
                the set size and ``C`` is the channel count.
            context_size: Optional split point along ``S``. When given, the
                inducing points only attend to ``x[..., :context_size, :]``,
                preventing target elements from leaking into the context. A
                value of ``0`` falls back to attending over the full set ``x``.
                Mutually exclusive with ``context_mask``.
            context_mask: Optional boolean mask with shape ``[S]`` selecting
                the context elements. Must be shared across the batch. Mutually
                exclusive with ``context_size``.
            seqused_key_value: Optional valid context lengths with shape
                ``[...]`` and dtype ``torch.int32``, applied when the inducing
                points attend to the context. Allows per-batch context sizes
                beyond the shared ``context_size``/``context_mask``.

        Returns:
            Tensor with shape ``[..., S, C]``.
        """
        if context_size is not None and context_mask is not None:
            raise ValueError(
                "Cannot pass both `context_size` and `context_mask`"
            )
        if context_mask is not None and context_mask.dtype != torch.bool:
            raise ValueError("`context_mask` must have dtype torch.bool")

        key_value = x  # [..., KV, C]
        if context_size is not None:
            key_value = x[..., :context_size, :]
        elif context_mask is not None:
            key_value = x[..., context_mask, :]

        # When the restricted context is empty, fall back to the full set `x`
        # and drop the (now invalid) length restriction.
        if key_value.size(-2) == 0:
            key_value = x  # [..., S, C]
            seqused_key_value = None

        hidden = self.transformer_1(
            query=self.inducing_points,  # [M, C]
            key_value=key_value,  # [..., KV, C]
            seqused_key_value=seqused_key_value,  # [...]
        )  # [..., M, C]
        return self.transformer_2(
            query=x,  # [..., S, C]
            key_value=hidden,  # [..., M, C]
        )  # [..., S, C]


class SetTransformer(torch.nn.Module):
    r"""Set Transformer encoder: a stack of induced self-attention blocks.

    Each :class:`InducedSelfAttentionBlock` processes the full set while only
    letting the inducing points attend to the context portion, yielding a
    permutation-equivariant, linear-time encoder over variable-sized sets.

    Introduced in `"Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_.

    Args:
        channels: Input and output channel count.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of the MLP in each block.
        num_layers: Number of induced self-attention blocks in the stack.
        num_inducing_points: Number of learned inducing points per block.
        qassmax: Whether to use :class:`~sdm.nn.QASSMax`. Applied only to the
            first attention of each
            :class:`~sdm.nn.InducedSelfAttentionBlock`, where the inducing
            points aggregate information from the variable-length context.
        device: The device to use for module parameters.
        dtype: The dtype to use for module parameters.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        feedforward_channels: int,
        num_layers: int,
        num_inducing_points: int = 16,
        qassmax: bool = False,
        norm_bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.blocks = ModuleList(
            InducedSelfAttentionBlock(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                num_inducing_points=num_inducing_points,
                qassmax=qassmax,
                norm_bias=norm_bias,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        x: Tensor,  # [..., S, C]
        context_size: int | None = None,
        context_mask: Tensor | None = None,  # [S]
        seqused_key_value: Tensor | None = None,  # [...]
    ) -> Tensor:  # [..., S, C]
        r"""Forward pass of the Set Transformer encoder.

        Args:
            x: Input set elements with shape ``[..., S, C]``, where ``S`` is
                the set size and ``C`` is the channel count.
            context_size: Optional split point along ``S`` shared by all
                blocks. See :meth:`InducedSelfAttentionBlock.forward`. Mutually
                exclusive with ``context_mask``.
            context_mask: Optional boolean mask with shape ``[S]`` shared by
                all blocks. Mutually exclusive with ``context_size``.
            seqused_key_value: Optional valid context lengths with shape
                ``[...]`` and dtype ``torch.int32``, shared by all blocks. See
                :meth:`InducedSelfAttentionBlock.forward`.

        Returns:
            Tensor with shape ``[..., S, C]``.
        """
        for block in self.blocks:
            x = block(
                x,
                context_size=context_size,
                context_mask=context_mask,
                seqused_key_value=seqused_key_value,
            )
        return x
