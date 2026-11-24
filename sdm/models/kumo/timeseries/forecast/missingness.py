# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.nn import Attention


class AdaptiveRevIN(torch.nn.Module):
    prior: Tensor
    initialized: Tensor

    def __init__(
        self,
        num_variates: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "prior", torch.ones(num_variates, 1, device=device, dtype=dtype)
        )
        self.register_buffer("initialized", torch.tensor(False, device=device))

    def forward(
        self,
        x: Tensor,
        observed: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        observed = observed & valid.unsqueeze(-1)
        count = observed.sum(-1, keepdim=True)
        mean = x.masked_fill(~observed, 0).sum(
            -1, keepdim=True
        ) / count.clamp_min(1)
        var = (
            (x - mean).square().masked_fill(~observed, 0).sum(-1, keepdim=True)
        )
        var = var / count.clamp_min(1)
        scale = torch.where(count > 0, var.sqrt(), 1) + 1e-5
        mean, scale = mean.detach(), scale.detach()

        prior = torch.where(self.initialized, self.prior, 1)
        extra = x.size(-2) - prior.size(-2)
        if extra > 0:
            prior = torch.cat(
                [prior, prior.mean(-2, keepdim=True).expand(extra, 1)], -2
            )
        prior = prior[: x.size(-2)]
        rate = 1 - observed.sum((-2, -1)) / (valid.sum(-1) * x.size(-1))
        shrinkage = ((rate - 0.5) / 0.5).clamp(0, 0.9).to(x.dtype)
        shrinkage = shrinkage[..., None, None]
        return mean * (1 - shrinkage), scale * (
            1 - shrinkage
        ) + prior * shrinkage


class ObservationAttentionBias(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.log_decay = torch.nn.Parameter(
            torch.full((num_heads,), -2.0, **factory_kwargs)
        )
        self.obs_bonus = torch.nn.Parameter(
            torch.full((num_heads,), 0.3, **factory_kwargs)
        )
        self.miss_penalty = torch.nn.Parameter(
            torch.full((num_heads,), 0.3, **factory_kwargs)
        )

    def forward(self, observed: Tensor) -> Tensor:
        patches = observed.size(-1)
        groups = observed.long().cumsum(-1)
        counts = groups.new_zeros((*groups.shape[:-1], patches + 1))
        counts.scatter_add_(-1, groups, (~observed).long())
        gap = counts.gather(-1, groups) * ~observed
        confidence = observed * self.obs_bonus.mean()
        confidence = confidence - ~observed * self.miss_penalty.mean() * (
            gap / 32
        ).clamp(0, 1)
        pos = torch.arange(patches, device=observed.device)
        distance = (pos[:, None] - pos[None, :]).abs()
        temporal = -self.log_decay.exp().clamp_max(5)[:, None, None] * distance
        return (temporal + confidence[..., None, None, :]).clamp(-10, 10)


class CrossChannelAttention(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_variates: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.attn = Attention(channels, num_heads, **factory_kwargs)
        self.norm = torch.nn.LayerNorm(channels, **factory_kwargs)
        self.channel_embed = torch.nn.Embedding(
            num_variates, channels, **factory_kwargs
        )
        self.residual_gate = torch.nn.Parameter(
            torch.tensor(-3.0, **factory_kwargs)
        )

    def forward(self, x: Tensor, observed: Tensor, valid: Tensor) -> Tensor:
        ids = torch.arange(x.size(-3), device=x.device)
        ids = ids.clamp_max(self.channel_embed.num_embeddings - 1)
        hidden = x.transpose(-3, -2) + self.channel_embed(
            ids
        )  # [..., P, C, D]
        keys = observed.transpose(-2, -1).clone()  # [..., P, C]
        keys[..., 0] |= ~keys.any(-1)
        mixed = self.attn(hidden, attn_mask=keys[..., None, None, :])
        mixed = self.norm(hidden + self.residual_gate.sigmoid() * mixed)
        return torch.where(
            valid[..., :, None, None], mixed.transpose(-3, -2), x
        )


class MissingnessAdapter(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        hidden_channels = max(16, channels // 4)
        self.mask_proj = torch.nn.Linear(2, channels, **factory_kwargs)
        self.norm = torch.nn.LayerNorm(channels, **factory_kwargs)
        self.down = torch.nn.Linear(
            channels, hidden_channels, **factory_kwargs
        )
        self.up = torch.nn.Linear(hidden_channels, channels, **factory_kwargs)
        self.gate_logit = torch.nn.Parameter(torch.zeros((), **factory_kwargs))
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor, observed: Tensor, valid: Tensor) -> Tensor:
        missing = (~observed & valid.unsqueeze(-1)).to(x.dtype)
        fraction = missing.sum(-2, keepdim=True) / valid.sum(-1)[
            ..., None, None
        ].clamp_min(1)
        features = torch.stack([missing, fraction.expand_as(missing)], -1)
        hidden = self.norm(x + self.mask_proj(features))
        residual = self.up(F.gelu(self.down(hidden)))
        gate = fraction.unsqueeze(-1) * self.gate_logit.sigmoid()
        return x + residual * gate * valid[..., :, None, None]
