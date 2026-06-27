r"""Set-transformer modules for structured tensor models."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import Parameter

from sdm.nn import TransformerBlock


class InducedTransformerBlock(torch.nn.Module):
    r"""Induced Transformer block from the Set Transformer.

    Introduced in the `"Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper, the block routes attention
    through a small set of ``M`` learned inducing points, reducing the cost of
    attending a query of size ``Q`` to a key/value context of size ``KV`` from
    :math:`O(Q \cdot KV)` to :math:`O((Q + KV) \cdot M)`:

    .. math::

        H = \mathrm{TransformerBlock}_1(I, \mathrm{key\_value}), \quad
        \mathrm{out} = \mathrm{TransformerBlock}_2(\mathrm{query}, H),

    where :math:`I` are the learned inducing points and :math:`H` are the
    inducing points after attending to the key/value elements. Passing
    ``key_value=None`` recovers the induced self-attention block (ISAB).

    Args:
        channels: Input and output channel count.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of the MLP in each block.
        num_inducing_points: Number of learned inducing points ``M``.
        qassmax: Whether to use :class:`~sdm.nn.QASSMax`. Applied only to the
            first attention, where the inducing points aggregate information
            from the variable-length key/value context.
        norm_bias: Whether LayerNorm uses learnable bias.
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
        query: Tensor,  # [..., Q, C]
        key_value: Tensor | None = None,  # [..., KV, C]
        seqused_key_value: Tensor | None = None,  # [...]
    ) -> Tensor:  # [..., Q, C]
        r"""Forward pass of the induced transformer block.

        Args:
            query: Query-side hidden states with shape ``[..., Q, C]``, where
                ``Q`` is the query set size and ``C`` is the channel count.
            key_value: Optional key/value-side context states with shape
                ``[..., KV, C]``. If omitted, ``query`` is used, recovering
                induced self-attention.
            seqused_key_value: Optional valid key/value lengths with shape
                ``[...]`` and dtype ``torch.int32``, applied when the inducing
                points attend to the key/value context.

        Returns:
            Tensor with shape ``[..., Q, C]``.
        """
        if key_value is None:
            key_value = query

        hidden = self.transformer_1(
            query=self.inducing_points,  # [M, C]
            key_value=key_value,  # [..., KV, C]
            seqused_key_value=seqused_key_value,  # [...]
        )  # [..., M, C]
        return self.transformer_2(
            query=query,  # [..., Q, C]
            key_value=hidden,  # [..., M, C]
        )  # [..., Q, C]
