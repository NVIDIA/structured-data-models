# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch-based time-series embeddings."""

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear, Parameter


def patch_mask(mask: Tensor, patch_len: int, stride: int) -> Tensor:
    """Convert a sequence mask into a patch mask.

    Args:
        mask: Sequence mask with shape ``[..., T]``.
        patch_len: Number of time steps in each patch.
        stride: Number of time steps between adjacent patches.

    Returns:
        Boolean mask with shape ``[..., P]``. A patch is valid only when all
        of its time steps are valid.
    """
    return mask.unfold(-1, patch_len, stride).bool().all(dim=-1)


class Patching(torch.nn.Module):
    """Create sliding time-series patches.

    Args:
        patch_len: Number of time steps in each patch.
        stride: Number of time steps between adjacent patches.
    """

    def __init__(self, patch_len: int, stride: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Create patches along the time dimension.

        Args:
            x: Time-series tensor with shape ``[..., T]``.

        Returns:
            Tensor with shape ``[..., P, L]``, where ``L`` is ``patch_len``.
        """
        return x.unfold(-1, self.patch_len, self.stride)


class PositionalEmbedding(torch.nn.Module):
    """Fixed sinusoidal position embeddings.

    Args:
        channels: Number of embedding channels.
        max_len: Maximum number of patch positions.
        device: The device.
        dtype: The dtype.
    """

    pe: Tensor

    def __init__(
        self,
        channels: int,
        max_len: int = 5000,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        position = torch.arange(max_len, **factory_kwargs).unsqueeze(-1)
        exponent = torch.arange(0, channels, 2, **factory_kwargs)
        exponent = exponent * (-math.log(10000.0) / channels)
        frequency = exponent.exp()

        embedding = torch.zeros(max_len, channels, **factory_kwargs)
        embedding[:, 0::2] = (position * frequency).sin()
        embedding[:, 1::2] = (position * frequency[: channels // 2]).cos()
        self.register_buffer("pe", embedding.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Return position embeddings for ``x``.

        Args:
            x: Patch embeddings with shape ``[..., P, C]``.

        Returns:
            Position embeddings with shape ``[1, P, C]``.
        """
        return self.pe[:, : x.size(-2)]


class PatchEmbedding(torch.nn.Module):
    """Embed time-series patches and replace missing patches by a token.

    Args:
        patch_len: Number of time steps in each patch.
        stride: Number of time steps between adjacent patches.
        channels: Number of embedding channels.
        dropout: Dropout probability.
        add_positional_embedding: Whether to add sinusoidal positions.
        bias: Whether the value projection learns an additive bias.
        orthogonal_gain: Gain used for orthogonal initialization. ``None``
            keeps the default linear-layer initialization.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        patch_len: int,
        stride: int,
        channels: int,
        dropout: float = 0.1,
        add_positional_embedding: bool = True,
        bias: bool = False,
        orthogonal_gain: float | None = 1.41,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = Linear(
            in_features=patch_len,
            out_features=channels,
            bias=bias,
            **factory_kwargs,
        )
        self.mask_embedding = Parameter(
            torch.zeros(channels, **factory_kwargs)
        )
        self.position_embedding = (
            PositionalEmbedding(channels, device=device, dtype=dtype)
            if add_positional_embedding
            else None
        )
        self.dropout = torch.nn.Dropout(dropout)

        if orthogonal_gain is not None:
            torch.nn.init.orthogonal_(
                self.value_embedding.weight,
                gain=orthogonal_gain,
            )
            if self.value_embedding.bias is not None:
                torch.nn.init.zeros_(self.value_embedding.bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Embed time-series patches.

        Args:
            x: Patched time series with shape ``[B, V, P, L]``.
            mask: Sequence mask with shape ``[B, T]``. ``None`` marks every
                patch as observed.

        Returns:
            Patch embeddings with shape ``[B, V, P, C]``.
        """
        out = self.value_embedding(x)
        if mask is not None:
            observed = patch_mask(mask, self.patch_len, self.stride)
            observed = observed.unsqueeze(1).unsqueeze(-1)
            out = torch.where(observed, out, self.mask_embedding)

        if self.position_embedding is not None:
            out = out + self.position_embedding(out)
        return self.dropout(out)
