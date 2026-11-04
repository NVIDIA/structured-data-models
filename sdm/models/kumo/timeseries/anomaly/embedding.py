# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


class DiffusionEmbedding(torch.nn.Module):
    """Embed discrete Kumo-Anomaly diffusion steps.

    The sinusoidal lookup is computed in float32 before conversion to the
    requested device and dtype.

    Args:
        num_steps: Number of diffusion steps in the lookup table.
        channels: Even sinusoidal embedding width, at least four.
        out_channels: Projection width. Defaults to ``channels``.
        device: The device.
        dtype: The parameter and output dtype.
    """

    embedding: Tensor

    def __init__(
        self,
        num_steps: int,
        channels: int = 128,
        out_channels: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if channels < 4 or channels % 2:
            raise ValueError("'channels' must be even and at least four")
        out_channels = channels if out_channels is None else out_channels
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.projection1 = torch.nn.Linear(
            in_features=channels, out_features=out_channels, **factory_kwargs
        )
        self.projection2 = torch.nn.Linear(
            in_features=out_channels,
            out_features=out_channels,
            **factory_kwargs,
        )
        torch.nn.init.xavier_uniform_(self.projection1.weight, gain=0.5)
        torch.nn.init.xavier_uniform_(self.projection2.weight, gain=0.5)

        # Compute phases in float32 on CPU to preserve upstream lookup values.
        steps = torch.arange(num_steps, device="cpu", dtype=torch.float32)
        frequencies = torch.arange(
            channels // 2, device="cpu", dtype=torch.float32
        )
        frequencies = 10.0 ** (frequencies / (channels // 2 - 1) * 4.0)
        angles = steps[:, None] * frequencies[None, :]
        self.register_buffer(
            "embedding",
            torch.cat([angles.sin(), angles.cos()], dim=-1).to(
                device=self.projection1.weight.device,
                dtype=self.projection1.weight.dtype,
            ),
            persistent=False,
        )

    def forward(self, step: Tensor) -> Tensor:
        """Embed discrete diffusion steps.

        Args:
            step: Integer indices with shape ``[...]`` in ``[0, num_steps)``.

        Returns:
            Embeddings with shape ``[..., C]``, where ``C`` is ``out_channels``
            (defaults to ``channels``).
        """
        x = F.silu(self.projection1(self.embedding[step]))
        return F.silu(self.projection2(x))


class SideEmbedding(torch.nn.Module):
    """Embed time coordinates and feature identities for Kumo-Anomaly.

    Sinusoidal time embeddings and learned feature embeddings are normalized
    separately before concatenation. The unconditional model also appends the
    conditioning mask, matching Kumo-TS. Time phases are computed in float32
    before conversion to the parameter dtype.

    Args:
        in_channels: Number of features in checkpoint order.
        time_channels: Even sinusoidal time embedding width, at least two.
        feature_channels: Learned feature embedding width.
        is_unconditional: Append the conditioning mask as the last channel.
        device: The device.
        dtype: The parameter and output dtype.
    """

    def __init__(
        self,
        in_channels: int,
        time_channels: int = 128,
        feature_channels: int = 16,
        is_unconditional: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.time_channels = time_channels
        self.is_unconditional = is_unconditional
        self.embed_layer = torch.nn.Embedding(
            num_embeddings=in_channels,
            embedding_dim=feature_channels,
            **factory_kwargs,
        )
        self.embed_norm = torch.nn.LayerNorm(
            normalized_shape=feature_channels,
            **factory_kwargs,
        )
        self.time_embed_norm = torch.nn.LayerNorm(
            normalized_shape=time_channels,
            **factory_kwargs,
        )

    def forward(self, time: Tensor, mask: Tensor) -> Tensor:
        """Build side information without changing time or feature order.

        Args:
            time: Model time coordinates with shape ``[B, L]``. Coordinates
                are used as supplied, with no normalization or resampling.
            mask: Conditioning mask with shape ``[B, K, L]`` in checkpoint
                feature order, where ``K`` equals ``in_channels``. One marks
                an observed value used for conditioning; zero marks a target.
                Both inputs must be on the module's device.

        Returns:
            Side information with shape ``[B, S, K, L]`` and parameter dtype.
            ``S`` is ``time_channels + feature_channels``, plus one when
            ``is_unconditional`` is set.
        """
        batch, features, length = mask.shape
        frequencies = 1 / torch.pow(
            self=10000.0,
            exponent=torch.arange(
                start=0,
                end=self.time_channels,
                step=2,
                device=time.device,
                dtype=torch.float32,
            )
            / self.time_channels,
        )
        angles = time.float().unsqueeze(-1) * frequencies
        time_embed = torch.stack([angles.sin(), angles.cos()], dim=-1)
        time_embed = self.time_embed_norm(
            time_embed.flatten(-2).to(self.embed_layer.weight.dtype)
        )
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, features, -1)
        feature_embed = self.embed_norm(self.embed_layer.weight)
        feature_embed = feature_embed[None, None].expand(batch, length, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)  # [B, S, K, L]
        if self.is_unconditional:
            side_info = torch.cat(
                tensors=[side_info, mask.unsqueeze(1).to(side_info.dtype)],
                dim=1,
            )
        return side_info
