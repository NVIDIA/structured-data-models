# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Released Kumo-TS missingness-aware inference components.

Adapted from Kumo-TS v1.0.1 (30bbf68), Kumo-Forecast/impute/model.py.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.models.kumo.timeseries.forecast.icl import ICLBlock
from sdm.nn import Attention, SinusoidalEmbedding


class AdaptiveRevIN(torch.nn.Module):
    """Observed-value statistics shrunk toward the checkpoint's scale prior."""

    prior: Tensor
    initialized: Tensor

    def __init__(
        self,
        num_variates: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "prior", torch.ones(num_variates, 1, device=device)
        )
        self.register_buffer("initialized", torch.tensor(False, device=device))

    def forward(
        self,
        x: Tensor,
        observed: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return per-series mean and scale using a frozen prior."""
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
        shrinkage = ((rate - 0.5) / 0.5).clamp(0, 0.9)[..., None, None]
        return mean * (1 - shrinkage), scale * (
            1 - shrinkage
        ) + prior * shrinkage


class ObservationAttentionBias(torch.nn.Module):
    """Temporal distance and observation confidence bias for patches."""

    def __init__(
        self,
        num_heads: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.log_decay = torch.nn.Parameter(
            torch.full((num_heads,), -2.0, device=device)
        )
        self.obs_bonus = torch.nn.Parameter(
            torch.full((num_heads,), 0.3, device=device)
        )
        self.miss_penalty = torch.nn.Parameter(
            torch.full((num_heads,), 0.3, device=device)
        )

    def forward(self, observed: Tensor) -> Tensor:
        """Return additive attention bias with shape ``[..., H, P, P]``."""
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
    """Gated attention with channel identities and masks for missing keys."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_variates: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.attn = Attention(channels, num_heads, device=device)
        self.norm = torch.nn.LayerNorm(channels, device=device)
        self.channel_embed = torch.nn.Embedding(
            num_variates, channels, device=device
        )
        self.residual_gate = torch.nn.Parameter(
            torch.tensor(-3.0, device=device)
        )

    def forward(self, x: Tensor, observed: Tensor, valid: Tensor) -> Tensor:
        """Mix observed channels at each patch and preserve padding."""
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
    """Mask-conditioned residual with the released checkpoint's 'any' gate."""

    def __init__(
        self,
        channels: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        hidden_channels = max(16, channels // 4)
        self.mask_proj = torch.nn.Linear(2, channels, device=device)
        self.norm = torch.nn.LayerNorm(channels, device=device)
        self.down = torch.nn.Linear(channels, hidden_channels, device=device)
        self.up = torch.nn.Linear(hidden_channels, channels, device=device)
        self.gate_logit = torch.nn.Parameter(torch.zeros((), device=device))
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor, observed: Tensor, valid: Tensor) -> Tensor:
        """Apply the residual only where real channels have missing patches."""
        missing = (~observed & valid.unsqueeze(-1)).to(x.dtype)
        fraction = missing.sum(-2, keepdim=True) / valid.sum(-1)[
            ..., None, None
        ].clamp_min(1)
        features = torch.stack([missing, fraction.expand_as(missing)], -1)
        hidden = self.norm(x + self.mask_proj(features))
        residual = self.up(F.gelu(self.down(hidden)))
        gate = fraction.unsqueeze(-1) * self.gate_logit.sigmoid()
        return x + residual * gate * valid[..., :, None, None]


class MissingnessBackbone(torch.nn.Module):
    """Patch encoder and 24-step head for the released 96-step checkpoint."""

    context_size = 96
    prediction_size = 24
    patch_size = 8

    def __init__(
        self,
        channels: int = 1024,
        hidden_channels: int = 2816,
        num_layers: int = 24,
        num_heads: int = 16,
        cross_channel_heads: int = 8,
        num_variates: int = 7,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.normalizer = AdaptiveRevIN(num_variates, device=device)
        self.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, 1, device=device)
        )
        self.mask_embedding = torch.nn.Parameter(
            torch.zeros(channels, device=device)
        )
        self.lin = torch.nn.Linear(
            self.patch_size, channels, bias=False, device=device
        )
        self.pe = SinusoidalEmbedding()
        self.obs_attn_bias = ObservationAttentionBias(num_heads, device=device)
        self.icl = ICLBlock(
            channels=channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_buckets=32,
            max_distance=128,
            device=device,
        )
        self.cross_channel = CrossChannelAttention(
            channels=channels,
            num_heads=cross_channel_heads,
            num_variates=num_variates,
            device=device,
        )
        self.missingness_adapter = MissingnessAdapter(channels, device=device)
        self.head = torch.nn.Linear(
            self.context_size // self.patch_size * channels,
            self.prediction_size,
            device=device,
        )

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        """Forecast scaled histories ``[..., C, 96]`` with nonfinite gaps."""
        observed = x.isfinite() & valid.unsqueeze(-1)
        x = x.masked_fill(~observed, 0)
        mean, scale = self.normalizer(x, observed, valid)
        x = (
            ((x - mean) / scale)
            .nan_to_num(nan=0, posinf=0, neginf=0)
            .clamp(-10, 10)
        )
        x = x + ~observed * self.mask_token.reshape(())
        patches = x.unflatten(-1, (-1, self.patch_size))
        patch_observed = observed.unflatten(-1, (-1, self.patch_size)).any(-1)
        x = self.lin(patches)
        x = torch.where(patch_observed.unsqueeze(-1), x, self.mask_embedding)
        x = self.pe(x)
        bias = self.obs_attn_bias(patch_observed).to(x.dtype)
        # T5 uses finite minimum logits even for entirely missing channels.
        bias = bias.masked_fill(
            ~patch_observed[..., None, None, :], torch.finfo(x.dtype).min
        )
        x = self.icl(x, attn_mask=bias)
        x = self.cross_channel(x, patch_observed, valid)
        x = self.missingness_adapter(x, patch_observed, valid)
        return self.head(x.flatten(-2)) * scale + mean
