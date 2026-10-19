# Copyright (c) 2026 Mistral AI
# Licensed under the Apache-2.0 License; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList, Sequential, Sigmoid

from sdm.nn import RMSNorm, RotaryEmbedding, SwiGLU, TransformerBlock


class TimestralTransformerBlock(torch.nn.Module):
    """Time and variate attention from
    `Timestral <https://github.com/mistralai/timestral/tree/a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff>`_.

    Each attention step has a pre-normalized residual, followed by one
    pre-normalized SwiGLU residual after the last attention step.

    Args:
        channels: Number of input and output channels.
        num_heads: Number of attention heads.
        hidden_channels: Hidden channels of the SwiGLU module.
        kind: Attention axes and their order.
        rope: Rotary embedding for time attention. Omitted for variate
            attention. ``None`` disables rotary embeddings.
        eps: Epsilon for RMS normalization.
        attn_bias: Learn additive biases in attention projections and gates.
        mlp_bias: Learn additive biases in the SwiGLU projections.
        gate: Apply sigmoid output gating to each attention step.
        device: The device.
        dtype: The dtype.
    """  # noqa: D205

    def __init__(
        self,
        channels: int,
        num_heads: int,
        hidden_channels: int,
        kind: Literal[
            "time", "variate", "time_variate", "variate_time"
        ] = "time",
        rope: RotaryEmbedding | None = None,
        eps: float = 1e-6,
        attn_bias: bool = False,
        mlp_bias: bool = False,
        gate: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.axes = {
            "time": (True,),
            "variate": (False,),
            "time_variate": (True, False),
            "variate_time": (False, True),
        }[kind]
        self.layers = ModuleList()
        for index, is_time in enumerate(self.axes):
            self.layers.append(
                TransformerBlock(
                    channels=channels,
                    num_query_heads=num_heads,
                    query_norm=RMSNorm(channels, eps=eps, **factory_kwargs),
                    query_transform=rope if is_time else None,
                    key_transform=rope if is_time else None,
                    gate=Sequential(
                        Linear(
                            channels,
                            channels,
                            bias=attn_bias,
                            **factory_kwargs,
                        ),
                        Sigmoid(),
                    )
                    if gate
                    else None,
                    mlp=Sequential(
                        RMSNorm(channels, eps=eps, **factory_kwargs),
                        SwiGLU(
                            channels,
                            hidden_channels,
                            bias=mlp_bias,
                            **factory_kwargs,
                        ),
                    )
                    if index == len(self.axes) - 1
                    else None,
                    bias=attn_bias,
                    **factory_kwargs,
                )
            )

    def forward(
        self,
        x: Tensor,
        time_mask: Tensor | None = None,
        variate_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        """Apply attention over packed variates and time patches.

        Args:
            x: Input with shape ``[R, N, C]``, where ``R`` is the number of
                packed variates, ``N`` is the number of time patches, and ``C``
                is the number of channels.
            time_mask: Boolean or additive mask shaped ``[R, H, N, N]`` or
                ``[H, N, N]``, where ``H`` is the number of heads. Singleton
                dimensions broadcast. ``None`` uses causal time attention.
            variate_mask: Boolean or additive mask shaped ``[N, H, R, R]`` or
                ``[H, R, R]``. Singleton dimensions broadcast. Callers packing
                independent series must mask attention between series.
                ``None`` allows all variates.
            pos: Time positions broadcastable to ``[R, N]``. ``None`` uses
                contiguous positions starting at zero.

        Returns:
            Tensor with shape ``[R, N, C]``.
        """
        for is_time, layer in zip(self.axes, self.layers):
            if not is_time:
                x = x.transpose(0, 1)
            x = layer(
                query=x,
                attn_mask=time_mask if is_time else variate_mask,
                query_pos=pos if is_time else None,
                is_causal=is_time and time_mask is None,
            )
            if not is_time:
                x = x.transpose(0, 1)
        return x
