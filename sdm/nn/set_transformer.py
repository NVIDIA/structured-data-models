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
        max_context_size: Optional cap on the number of context elements the
            inducing points attend to. If the context exceeds this size it is
            randomly subsampled; if ``None`` all context elements are used.
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
        max_context_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_inducing_points = num_inducing_points
        self.max_context_size = max_context_size

        self.transformer_1 = TransformerBlock(
            channels=channels,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            qassmax=qassmax,
            **factory_kwargs,
        )
        self.transformer_2 = TransformerBlock(
            channels=channels,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
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
        generator: torch.Generator | None = None,
    ) -> Tensor:  # [..., S, C]
        r"""Forward pass of the induced self-attention block.

        Args:
            x: Input set elements with shape ``[..., S, C]``, where ``S`` is
                the set size and ``C`` is the channel count.
            context_size: Optional split point along ``S``. When given, the
                inducing points only attend to ``x[..., :context_size, :]``,
                preventing target elements from leaking into the context. A
                value of ``0`` falls back to self-attention over the inducing
                points. Mutually exclusive with ``context_mask``.
            context_mask: Optional boolean mask with shape ``[S]`` selecting
                the context elements. Must be shared across the batch. Mutually
                exclusive with ``context_size``.
            generator: Optional generator controlling context subsampling when
                ``max_context_size`` is set.

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

        # When the context is empty, self-attend over the inducing points
        # instead.
        if key_value.size(-2) == 0:
            hidden = self.transformer_1(query=self.inducing_points)  # [M, C]
            return self.transformer_2(query=x, key_value=hidden)

        if (
            self.max_context_size is not None
            and key_value.size(-2) > self.max_context_size
        ):
            index = torch.randperm(
                key_value.size(-2),
                device=key_value.device,
                generator=generator,
            )[: self.max_context_size]
            key_value = key_value[..., index, :]

        hidden = self.transformer_1(
            query=self.inducing_points,  # [M, C]
            key_value=key_value,  # [..., KV, C]
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
        max_context_size: Optional cap on the number of context elements the
            inducing points attend to. If the context exceeds this size it is
            randomly subsampled; if ``None`` all context elements are used.
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
        max_context_size: int | None = None,
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
                max_context_size=max_context_size,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        x: Tensor,  # [..., S, C]
        context_size: int | None = None,
        context_mask: Tensor | None = None,  # [S]
        generator: torch.Generator | None = None,
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
            generator: Optional generator controlling context subsampling when
                ``max_context_size`` is set.

        Returns:
            Tensor with shape ``[..., S, C]``.
        """
        for block in self.blocks:
            x = block(
                x,
                context_size=context_size,
                context_mask=context_mask,
                generator=generator,
            )
        return x
